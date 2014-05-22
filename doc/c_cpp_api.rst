.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 14:59:05 2013

=========
 C++ API
=========

The C++ API of ``xbob.learn.activation`` allows users to leverage from
automatic converters for classes in :py:class:`xbob.learn.activation`.  To use
the C API, clients should first, include the header file
``<xbob.learn.activation/api.h>`` on their compilation units and then, make
sure to call once ``import_xbob_learn_activation()`` at their module
instantiation, as explained at the `Python manual
<http://docs.python.org/2/extending/extending.html#using-capsules>`_.

Here is a dummy C example showing how to include the header and where to call
the import function:

.. code-block:: c++

   #include <xbob.learn.activation/api.h>

   PyMODINIT_FUNC initclient(void) {

     PyObject* m Py_InitModule("client", ClientMethods);

     if (!m) return;

     // imports dependencies
     if (import_xbob_blitz() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     if (import_xbob_io_base() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     if (import_xbob_learn_activation() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     // imports xbob.learn.activation C-API
     import_xbob_learn_activation();

   }

.. note::

  The include directory can be discovered using
  :py:func:`xbob.learn.activation.get_include`.

Activation Functors
-------------------

.. cpp:type:: PyBobLearnActivationObject

   The pythonic object representation for a ``bob::machine::Activation``
   object. It is the base class of all activation functors available in
   |project|. In C/C++ code, we recommend you only manipulate objects like this
   to keep your code agnostic to the activation type being used.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        bob::machine::Activation* base;
      } PyBobLearnActivationObject;

   .. cpp:member:: bob::machine::Activation* base

      A pointer to the activation functor virtual implementation.


.. cpp:function:: int PyBobLearnActivation_Check(PyObject* o)

   Checks if the input object ``o`` is a ``PyBobLearnActivationObject``.
   Returns ``1`` if it is, and ``0`` otherwise.


.. cpp:function:: PyObject* PyBobLearnActivation_NewFromActivation(boost::shared_ptr<bob::machine::Activation> a)

   Constructs a new :c:type:`PyBobLearnActivationObject` starting from a shared
   pointer to a pre-allocated `bob::machine::Activation` instance. This API is
   available so that return values from actuall C++ machines can be mapped into
   Python. It is the sole way to build an object of type :py:class:`Activation`
   without recurring to the derived classes.

.. note::

   Other object definitions exist for each of the specializations for
   activation functors found in |project|. They are exported through the module
   C-API, but we don't recommend using them since you'd loose generality. In
   case you do absolutely need to use any of these derivations, they have all
   the same object configuration:

   .. code-block:: c++

      typedef struct {
        PyBobLearnActivationObject parent;
        bob::machine::<Subtype>Activation* base;
      } PyBobLearn<Subtype>ActivationObject;

   Presently, ``<Subtype>`` can be one of:

     * Identity
     * Linear
     * Logistic
     * HyperbolicTangent
     * MultipliedHyperbolicTangent

   Type objects are also named consistently like
   ``PyBobLearn<Subtype>Activation_Type``.

.. include:: links.rst
