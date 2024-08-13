#include <eigenpy/eigenpy.hpp>

#include "sobec/python.hpp"

BOOST_PYTHON_MODULE(sobec_pywrap) {
  namespace bp = boost::python;

  bp::import("pinocchio");
  bp::import("crocoddyl");
  // Enabling eigenpy support, i.e. numpy/eigen compatibility.
  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<Eigen::VectorXi>();
  sobec::python::exposeStdContainers();
  sobec::python::exposeResidualVelCollision();
  sobec::python::exposeResidualCoMVelocity();
  sobec::python::exposeResidualCenterOfPressure();
  sobec::python::exposeResidualCenterOfFriction();
  sobec::python::exposeResidualFeetCollision();
  sobec::python::exposeResidualFlyHigh();
  sobec::python::exposeResidualFlyAngle();
  sobec::python::exposeResidualDCMPosition();
  sobec::python::exposeResidual2DSurface();
  sobec::python::exposeActivationQuadRef();
  // sobec::python::exposeDesigner();
  // sobec::python::exposeHorizonManager();
  // sobec::python::exposeModelFactory();
  sobec::python::exposeIntegratedActionLPF();
  sobec::python::exposeStateLPF();
  // sobec::python::exposeWBC();
  // sobec::python::exposeWBCHorizon();
  // sobec::python::exposeFootTrajectory();
  sobec::python::exposeFlex();
  sobec::python::exposeOCPWalk();
  sobec::python::exposeMPCWalk();

  // sobec::newcontacts::python::exposeContact6D();
  // sobec::newcontacts::python::exposeContact3D();
  // sobec::newcontacts::python::exposeContact1D();
  // sobec::newcontacts::python::exposeMultipleContacts();
  // sobec::newcontacts::python::exposeDAMContactFwdDyn();
  // sobec::newcontacts::python::exposeResidualContactForce();
}
