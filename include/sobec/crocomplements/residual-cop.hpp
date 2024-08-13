///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SOBEC_RESIDUAL_COP_HPP_
#define SOBEC_RESIDUAL_COP_HPP_

#include <boost/smart_ptr/shared_ptr.hpp>
#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/core/utils/deprecate.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/data/multibody.hpp>
#include <crocoddyl/multibody/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/multibody/joint/joint-generic.hpp>
#include <pinocchio/spatial/fwd.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "crocoddyl/multibody/contacts/contact-pin.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "sobec/fwd.hpp"

namespace sobec {

using namespace crocoddyl;

/**
 * @brief COP residual
 *
 * residual = [ tau_y/f_z, -tau_x/fx ]
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelCenterOfPressureTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataCenterOfPressureTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  /**
   * @brief Initialize the residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] contact_name Name of the contact
   * @param[in] nu          Dimension of control vector
   * link is attached
   */
  ResidualModelCenterOfPressureTpl(boost::shared_ptr<StateMultibody> state,
                                   const std::string contact_name,
                                   const std::size_t nu);

  virtual ~ResidualModelCenterOfPressureTpl();

  /**
   * @brief Compute the cop.
   *
   * @param[in] data  residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  /**
   * @brief Compute the derivatives of residual
   *
   * @param[in] data  residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);

  virtual boost::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract *const data);

  /**
   * @brief Return the reference contact name
   */
  std::string get_name() const { return contact_name_; }
  
  /** @brief Set the reference contact name */
  DEPRECATED("Do not use set_name, instead create a new residual model", 
              void set_name(const std::string id) { contact_name_ = id; }
  )

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  std::string contact_name_;
};

template <typename _Scalar>
struct ResidualDataCenterOfPressureTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  ResidualDataCenterOfPressureTpl(Model<Scalar> *const model,
                                  DataCollectorAbstract *const data)
      : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorContactTpl<Scalar> *d =
        dynamic_cast<DataCollectorContactTpl<Scalar> *>(this->shared);
    if (d == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorContact");
    }
    const std::string name = model->get_name();
    const boost::shared_ptr<StateMultibody> &state =
        boost::static_pointer_cast<StateMultibody>(model->get_state());

    bool found_contact = false;
    for (auto &it : d->contacts->contacts) {
      if (it.first == name) {
        ContactDataTpl<Scalar> *contactData =
            dynamic_cast<ContactDataTpl<Scalar> *>(it.second.get());
        if (contactData != NULL) {
          if (contactData->type != pinocchio::ContactType::CONTACT_6D) {
            throw_pretty(
                "Domain error: contact should be 6d for COP with name " + name);
          }
          found_contact = true;
          this->contact = it.second;
          this->jMf = boost::allocate_shared<pinocchio::SE3Tpl<Scalar>>(
              Eigen::aligned_allocator<pinocchio::SE3Tpl<Scalar>>(), it.second->jMf);
          break;
        }
        throw_pretty(
            "Domain error: there isn't defined at least a 6d contact with name " +
            name);

        // The frame of contact should be asserted!
        // if (this->contact->get_type()!=pinocchio::WORLD)
        //   throw_pretty(
        //                "Domain error: contact should be defined in WORLD for COP " +
        //                frame_name);
        
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data with first joint being " +
                   name);
    }
  }
  boost::shared_ptr<ForceDataAbstractTpl<Scalar> > contact;
  boost::shared_ptr<pinocchio::SE3Tpl<Scalar> > jMf; // Placement of the contact frame with respect to the reference joint
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace sobec

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "sobec/crocomplements/residual-cop.hxx"

#endif  // SOBEC_RESIDUAL_COP_HPP_
