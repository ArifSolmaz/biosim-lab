/*---------------------------------------------------------------------------*\
  biosim-lab — acousticRadiationForce
  Stage 4 template. Lagrangian cloud force model that adds the Gor'kov primary
  acoustic radiation force to particles carried by DPMFoam.

  Physics
  -------
  For a particle of volume V in a time-harmonic field with complex pressure
  amplitude p and fluid velocity amplitude v = grad(p)/(i*omega*rho_f), the
  Gor'kov potential is

      U = V * ( (f1/2)*kappa_f*<p^2>  -  (3/4)*f2*rho_f*<v^2> )
      f1 = 1 - kappa_p/kappa_f
      f2 = 2*(rho_p - rho_f)/(2*rho_p + rho_f)

  and the force is F = -grad(U).

  Reference: Bruus, Lab Chip 12:1014 (2012), doi:10.1039/c2lc21068a.

  The complex pressure field is supplied by biosim-lab as two volScalarFields
  (pAcousticRe, pAcousticIm) written into the case's 0/ directory; this class
  reads them, forms U on the mesh once, and interpolates -grad(U) to each
  parcel position every step.

  Licence: MIT, as the rest of biosim-lab.
\*---------------------------------------------------------------------------*/

#ifndef acousticRadiationForce_H
#define acousticRadiationForce_H

#include "ParticleForce.H"
#include "volFields.H"
#include "interpolation.H"

namespace Foam
{

template<class CloudType>
class acousticRadiationForce
:
    public ParticleForce<CloudType>
{
    // Private data

        //- Name of the real part of the acoustic pressure field
        const word pReName_;

        //- Name of the imaginary part of the acoustic pressure field
        const word pImName_;

        //- Fluid density [kg/m^3]
        const scalar rhoF_;

        //- Fluid sound speed [m/s]
        const scalar cF_;

        //- Particle compressibility [1/Pa]
        const scalar kappaP_;

        //- Particle density [kg/m^3]
        const scalar rhoP_;

        //- Angular frequency [rad/s]
        const scalar omega_;

        //- Gor'kov potential per unit particle volume [J/m^3], built in cacheFields
        autoPtr<volScalarField> UGorkovPtr_;

        //- -grad(U) per unit particle volume [N/m^3]
        autoPtr<volVectorField> forcePerVolumePtr_;

        //- Interpolator for the force field
        autoPtr<interpolation<vector>> forceInterpPtr_;


public:

    //- Runtime type information
    TypeName("acousticRadiation");

    // Constructors

        acousticRadiationForce
        (
            CloudType& owner,
            const fvMesh& mesh,
            const dictionary& dict
        );

        acousticRadiationForce(const acousticRadiationForce& arf);

        virtual autoPtr<ParticleForce<CloudType>> clone() const
        {
            return autoPtr<ParticleForce<CloudType>>
            (
                new acousticRadiationForce<CloudType>(*this)
            );
        }

    //- Destructor
    virtual ~acousticRadiationForce();


    // Member Functions

        //- Cache the Gor'kov potential and its gradient (called once per step)
        virtual void cacheFields(const bool store);

        //- Calculate the non-coupled force
        virtual forceSuSp calcNonCoupled
        (
            const typename CloudType::parcelType& p,
            const typename CloudType::parcelType::trackingData& td,
            const scalar dt,
            const scalar mass,
            const scalar Re,
            const scalar muc
        ) const;
};

} // End namespace Foam

#ifdef NoRepository
    #include "acousticRadiationForce.C"
#endif

#endif
