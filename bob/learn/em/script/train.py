"""A script to help with EM training.
"""
import logging
import click
import subprocess
import os
import sys
import shutil
import time
import numpy as np
from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    log_parameters,
)
from bob.io.base import vstack_features
from gridtk.tools import get_array_job_slice
import bob.learn.em
from glob import glob
from bob.io.base import HDF5File
from ..train import _set_average

logger = logging.getLogger(__name__)
FINISHED = "finished.txt"
STATS = "stats.hdf5"
SLEEP = 5


@click.command(
    entry_point_group="bob.config",
    cls=ConfigCommand,
    epilog="""\b
Examples:
  $ bob em train -vvv config.py -o /tmp/gmm -- --array 64 --jman-options '-q q1d -i -m 4G' ...

Note: samples must be sorted!
""",
)
@click.option(
    "--reader",
    required=True,
    cls=ResourceOption,
    help="""A callable that takes a sample and returns the loaded sample. The reader "
    "must return exactly one sample and sample must be a float64 1D array.""",
)
@click.option(
    "--samples",
    required=True,
    cls=ResourceOption,
    help="A list of samples to be loaded with reader. The samples must be stable. "
    "The script will be called several times in separate processes. Each time the "
    "config file is loaded the samples should have the same order and must be exactly "
    "the same! It's best to sort them!",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    cls=ResourceOption,
    help="The directory to save the machines and statistics.",
)
@click.option(
    "--array",
    "-t",
    type=click.INT,
    required=True,
    cls=ResourceOption,
    help="The number of workers to create. All workers must run at the same time for "
    "this script to work.",
)
@click.option(
    "--trainer", cls=ResourceOption, required=True, help="The trainer to be used."
)
@click.option(
    "--machine", cls=ResourceOption, required=True, help="The machine to be used."
)
@click.option(
    "--max-iterations",
    type=click.INT,
    default=50,
    cls=ResourceOption,
    show_default=True,
    help="The maximum number of iterations to train a machine.",
)
@click.option(
    "--convergence-threshold",
    type=click.FLOAT,
    default=4e-5,
    show_default=True,
    cls=ResourceOption,
    help="The convergence threshold to train a machine. If None, the training "
    "procedure will stop with the iterations criteria.",
)
@click.option(
    "--initialization-stride",
    type=click.INT,
    default=1,
    show_default=True,
    cls=ResourceOption,
    help="The stride to use for selecting a subset of samples to initialize the "
    "machine. Must be 1 or greater.",
)
@click.option(
    "--jman-options",
    default=" ",
    show_default=True,
    cls=ResourceOption,
    help="Additional options to be given to jman",
)
@click.option(
    "--jman",
    default="jman",
    show_default=True,
    cls=ResourceOption,
    help="Path to the jman script.",
)
@click.option(
    "--step",
    hidden=True,
    help="Internal option. If this parameter is given, then this process is a worker.",
)
@verbosity_option(cls=ResourceOption)
@click.pass_context
def train(
    ctx,
    samples,
    reader,
    output_dir,
    array,
    trainer,
    machine,
    max_iterations,
    convergence_threshold,
    initialization_stride,
    jman_options,
    jman,
    step,
    **kwargs,
):
    """Trains Bob machines using bob.learn.em.

    To debug the E Step, run the script like this:
    SGE_TASK_ID=1 SGE_TASK_FIRST=1 SGE_TASK_STEPSIZE=1 SGE_TASK_LAST=1 bin/python -m IPython --pdb -- bin/bob em train -vvv config.py --step e
    """
    log_parameters(logger, ignore=("samples",))
    logger.debug("len(samples): %d", len(samples))
    trainer_type = trainer.__class__.__name__

    if step == "e":
        # slice into samples for this worker
        samples = samples[get_array_job_slice(len(samples))]
        if trainer_type != "KMeansTrainer":
            trainer.initialize(machine)
        return e_step(samples, reader, output_dir, trainer, machine)

    if hasattr(trainer, "finalize"):
        raise NotImplementedError

    os.makedirs(output_dir, exist_ok=True)
    if len(os.listdir(output_dir)) != 0:
        click.echo(
            f"The output_dir: {output_dir} is not empty! This will cause problems!"
        )
        raise click.Abort

    n_samples = len(samples)
    # some array jobs may not get any samples
    # for example if n_samples is 241 and array is 64,
    # each worker gets 4 samples and that means only 61 workers would get samples to
    # work with
    n_jobs = int(np.ceil(n_samples / np.ceil(n_samples / array)))

    # initialize
    if trainer_type in ("KMeansTrainer", "ML_GMMTrainer"):
        initilization_samples = samples[::initialization_stride]
        logger.info(
            "Loading %d samples to initialize the machine", len(initilization_samples)
        )
        data = read_samples(reader, initilization_samples)

        logger.info("Initializing the trainer (and maybe machine)")
        trainer.initialize(machine, data)

        if trainer_type == "ML_GMMTrainer":
            logger.info("Initializing GMM with KMeans.")
            kmeans = bob.learn.em.KMeansMachine(*machine.shape)
            kmeans.means = machine.means
            [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(
                data
            )
            machine.variances = variances
            machine.weights = weights
        del data
    else:
        trainer.initialize(machine)

    if hasattr(trainer, "reset_accumulators"):
        trainer.reset_accumulators(machine)

    # submit workers
    command = (
        [jman, "submit", "--print-id", "--array", str(array)]
        + jman_options.split()
        + ["--"]
        + sys.argv
        + ["--step", "e"]
    )
    click.echo("")
    logger.debug("Calling jman: %s", command)
    subprocess.call(command)
    logger.info("Submitted job array")

    # E step 0
    step = 0
    logger.info("Iteration = %d/%d", step, max_iterations)
    distributed_e_step(output_dir, trainer, machine, n_samples, step, n_jobs)
    average_output = 0
    average_output_previous = 0

    if hasattr(trainer, "compute_likelihood"):
        average_output = trainer.compute_likelihood(machine)
    for step in range(1, max_iterations):
        logger.info("Iteration = %d/%d", step, max_iterations)
        average_output_previous = average_output
        trainer.m_step(machine, None)
        distributed_e_step(output_dir, trainer, machine, n_samples, step, n_jobs)

        if hasattr(trainer, "compute_likelihood"):
            average_output = trainer.compute_likelihood(machine)

        if isinstance(machine, bob.learn.em.KMeansMachine):
            logger.info("average euclidean distance = %f", average_output)
        else:
            logger.info("log likelihood = %f", average_output)

        convergence_value = abs(
            (average_output_previous - average_output) / average_output_previous
        )
        logger.info("convergence value = %f", convergence_value)

        # Terminates if converged (and likelihood computation is set)
        if (
            convergence_threshold is not None
            and convergence_value <= convergence_threshold
        ):
            logger.info("Reached convergence_threshold.")
            break
    # do on more m_step since it's cheap and e step is already done
    trainer.m_step(machine, None)

    # save machine
    with HDF5File(os.path.join(output_dir, "machine.hdf5"), "w") as f:
        machine.save(f)

    path = os.path.join(output_dir, FINISHED)
    with open(path, "w") as f:
        f.write("finished!")


def workers_finished(fldr, n_jobs):
    stats = glob(os.path.join(fldr, f"*_{STATS}"))
    return len(stats) == n_jobs


def load_trainers(trainer, machine, fldr):
    stats = glob(os.path.join(fldr, f"*_{STATS}"))
    trainers = [load_statistics(trainer, machine, path) for path in stats]
    return trainers


def distributed_e_step(output_dir, trainer, machine, n_samples, step, n_jobs):
    # write machine for workers to pick it up
    fldr = os.path.join(output_dir, f"machine_{step}")
    os.makedirs(fldr, exist_ok=True)
    # atomic writing
    path = os.path.join(fldr, "machine_temp.hdf5")
    new_path = os.path.join(fldr, "machine.hdf5")
    with HDF5File(path, "w") as f:
        machine.save(f)
    shutil.move(path, new_path)
    # wait for all workers to write their stats
    while not workers_finished(fldr, n_jobs):
        logger.debug("Waiting for workers to finish.")
        time.sleep(SLEEP)
        continue
    # load all statistics
    trainers = load_trainers(trainer, machine, fldr)
    assert len(trainers) == n_jobs, len(trainers)
    # average E step
    trainer_type = trainer.__class__.__name__
    _set_average(trainer, trainers, machine, n_samples, trainer_type)
    # clean up the folder
    shutil.rmtree(fldr)


def read_evaluated(output_dir, sge_task_id):
    path = os.path.join(output_dir, f"evaluated_{sge_task_id}.txt")
    if not os.path.isfile(path):
        return []
    evaluated = []
    with open(path) as f:
        for line in f:
            evaluated.append(line.strip())
    return evaluated


def save_evaluated(output_dir, sge_task_id, evaluated):
    path = os.path.join(output_dir, f"evaluated_{sge_task_id}.txt")
    with open(path, "w") as f:
        for step in evaluated:
            f.write(f"{step}\n")


def finished(output_dir):
    path = os.path.join(output_dir, FINISHED)
    return os.path.isfile(path)


def get_step(machine_path):
    machine_path = os.path.dirname(machine_path)
    machine_path = os.path.basename(machine_path)
    return machine_path.split("_")[-1]


def find_all_machines(output_dir):
    machines = glob(os.path.join(output_dir, "machine_*", "machine.hdf5"))
    # integer sort according to steps
    machines = sorted(machines, key=lambda x: int(get_step(x)))
    machines = {get_step(x): x for x in machines}
    return machines


def return_new_machine(output_dir, evaluated, machine):
    machines = find_all_machines(output_dir)
    for step, machine_path in machines.items():
        if step not in evaluated:
            with HDF5File(machine_path, "r") as f:
                return step, machine.__class__(f)
    return None, None


def save_statistics(trainer, data, step, output_dir, sge_task_id):
    output_path = os.path.join(output_dir, f"machine_{step}", f"{sge_task_id}_{STATS}")
    logger.debug("Saving statistics to: %s", output_path)
    trainer_type = trainer.__class__.__name__
    temp_path = output_path + ".hdf5"
    with HDF5File(temp_path, "w") as f:

        if trainer_type == "KMeansTrainer":
            f["zeroeth_order_statistics"] = trainer.zeroeth_order_statistics
            f["first_order_statistics"] = trainer.first_order_statistics
            f["average_min_distance"] = trainer.average_min_distance * len(data)

        elif trainer_type in ("ML_GMMTrainer", "MAP_GMMTrainer"):
            trainer.gmm_statistics.save(f)

        elif trainer_type == "IVectorTrainer":
            f["acc_fnormij_wij"] = trainer.acc_fnormij_wij
            f["acc_nij_wij2"] = trainer.acc_nij_wij2
            f["acc_nij"] = trainer.acc_nij
            f["acc_snormij"] = trainer.acc_snormij
    shutil.move(temp_path, output_path)


def load_statistics(trainer, machine, path):
    # duplicate trainer
    trainer = trainer.__class__(trainer)
    if hasattr(trainer, "reset_accumulators"):
        # some trainers have issues and need this
        trainer.reset_accumulators(machine)
    trainer_type = trainer.__class__.__name__
    with HDF5File(path, "r") as f:

        if trainer_type == "KMeansTrainer":
            zeros = f["zeroeth_order_statistics"]
            trainer.zeroeth_order_statistics = np.array(zeros).reshape((-1,))
            trainer.first_order_statistics = f["first_order_statistics"]
            trainer.average_min_distance = f["average_min_distance"]

        elif trainer_type in ("ML_GMMTrainer", "MAP_GMMTrainer"):
            trainer.gmm_statistics.load(f)

        elif trainer_type == "IVectorTrainer":
            trainer.acc_fnormij_wij = f["acc_fnormij_wij"]
            trainer.acc_nij_wij2 = f["acc_nij_wij2"]
            trainer.acc_nij = f["acc_nij"]
            trainer.acc_snormij = f["acc_snormij"]

    return trainer


def e_step(samples, reader, output_dir, trainer, machine):
    if len(samples) == 0:
        print("This worker did not get any samples.")
        return
    logger.info("Loading %d samples", len(samples))
    data = read_samples(reader, samples)
    logger.info("Loaded all samples")
    sge_task_id = os.environ["SGE_TASK_ID"]
    while not finished(output_dir):
        # check which machines we have evaluated
        evaluated = read_evaluated(output_dir, sge_task_id)
        # check if new machines exist
        step, _ = return_new_machine(output_dir, evaluated, machine)
        if step is None:
            logger.debug("Waiting for another machine to appear.")
            time.sleep(SLEEP)
            continue
        step, machine = return_new_machine(output_dir, evaluated, machine)
        assert step is not None
        # run E step
        bob.learn.em.train(trainer, machine, data, max_iterations=0, initialize=False)
        # save accumulated statistics
        save_statistics(trainer, data, step, output_dir, sge_task_id)
        # update evaluated
        evaluated.append(step)
        save_evaluated(output_dir, sge_task_id, evaluated)


def read_samples(reader, samples):
    # read one sample to see if data is numpy arrays
    data = reader(samples[0])
    if isinstance(data, np.ndarray):
        samples = vstack_features(reader, samples, same_size=False)
    else:
        samples = [reader(s) for s in samples]
    return samples
