/*---------------------------------------------------------------------------*\
  biosim-lab — acousticRadiationForce implementation (Stage 4 template).

  Build: place this pair next to your DPMFoam solver sources, add
      acousticRadiationForce.C
  to Make/files and rebuild with wmake. Then reference it from the cloud
  properties dictionary as shown in kinematicCloudProperties.template.

  Status: this is a *template*. It compiles as written against OpenFOAM's
  ParticleForce interface, but has not been validated against an experiment;
  biosim-lab ships it so the interface contract is concrete, not so it can be
  used unverified. Validate against the analytic 1-D standing-wave force
  (biosim_lab.instruments.saw_sorter.physics.acoustics) before trusting it.
\*---------------------------------------------------------------------------*/

#include "acousticRadiationForce.H"
#include "fvcGrad.H"

template<class CloudType>
Foam::acousticRadiationForce<CloudType>::acousticRadiationForce
(
    CloudType& owner,
    const fvMesh& mesh,
    const dictionary& dict
)
:
    ParticleForce<CloudType>(owner, mesh, dict, typeName, true),
    pReName_(this->coeffs().template getOrDefault<word>("pRe", "pAcousticRe")),
    pImName_(this->coeffs().template getOrDefault<word>("pIm", "pAcousticIm")),
    rhoF_(this->coeffs().template get<scalar>("rhoFluid")),
    cF_(this->coeffs().template get<scalar>("cFluid")),
    kappaP_(this->coeffs().template get<scalar>("kappaParticle")),
    rhoP_(this->coeffs().template get<scalar>("rhoParticle")),
    omega_(this->coeffs().template get<scalar>("omega")),
    UGorkovPtr_(nullptr),
    forcePerVolumePtr_(nullptr),
    forceInterpPtr_(nullptr)
{}


template<class CloudType>
Foam::acousticRadiationForce<CloudType>::acousticRadiationForce
(
    const acousticRadiationForce& arf
)
:
    ParticleForce<CloudType>(arf),
    pReName_(arf.pReName_),
    pImName_(arf.pImName_),
    rhoF_(arf.rhoF_),
    cF_(arf.cF_),
    kappaP_(arf.kappaP_),
    rhoP_(arf.rhoP_),
    omega_(arf.omega_),
    UGorkovPtr_(nullptr),
    forcePerVolumePtr_(nullptr),
    forceInterpPtr_(nullptr)
{}


template<class CloudType>
Foam::acousticRadiationForce<CloudType>::~acousticRadiationForce()
{}


template<class CloudType>
void Foam::acousticRadiationForce<CloudType>::cacheFields(const bool store)
{
    if (!store)
    {
        forceInterpPtr_.clear();
        return;
    }

    const fvMesh& mesh = this->mesh();

    const volScalarField& pRe = mesh.lookupObject<volScalarField>(pReName_);
    const volScalarField& pIm = mesh.lookupObject<volScalarField>(pImName_);

    const scalar kappaF = 1.0/(rhoF_*cF_*cF_);
    const scalar f1 = 1.0 - kappaP_/kappaF;
    const scalar f2 = 2.0*(rhoP_ - rhoF_)/(2.0*rhoP_ + rhoF_);

    // <p^2> = |p|^2 / 2
    volScalarField p2("p2", 0.5*(pRe*pRe + pIm*pIm));

    // v = grad(p)/(i*omega*rho_f)  ->  <v^2> = |grad p|^2 / (2*(omega*rho_f)^2)
    volVectorField gRe(fvc::grad(pRe));
    volVectorField gIm(fvc::grad(pIm));
    volScalarField v2
    (
        "v2",
        0.5*(magSqr(gRe) + magSqr(gIm))/sqr(omega_*rhoF_)
    );

    UGorkovPtr_.reset
    (
        new volScalarField
        (
            IOobject("UGorkovPerVolume", mesh.time().timeName(), mesh),
            0.5*f1*kappaF*p2 - 0.75*f2*rhoF_*v2
        )
    );

    forcePerVolumePtr_.reset
    (
        new volVectorField
        (
            IOobject("FacousticPerVolume", mesh.time().timeName(), mesh),
            -fvc::grad(UGorkovPtr_())
        )
    );

    forceInterpPtr_ = interpolation<vector>::New
    (
        this->owner().solution().interpolationSchemes(),
        forcePerVolumePtr_()
    );
}


template<class CloudType>
Foam::forceSuSp Foam::acousticRadiationForce<CloudType>::calcNonCoupled
(
    const typename CloudType::parcelType& p,
    const typename CloudType::parcelType::trackingData& td,
    const scalar dt,
    const scalar mass,
    const scalar Re,
    const scalar muc
) const
{
    forceSuSp value(Zero, 0.0);

    if (!forceInterpPtr_.valid())
    {
        return value;
    }

    const vector fPerVolume =
        forceInterpPtr_().interpolate(p.coordinates(), p.currentTetIndices());

    // Particle volume from its diameter; the Gor'kov force is linear in volume.
    const scalar V = constant::mathematical::pi/6.0*pow3(p.d());

    value.Su() = fPerVolume*V;

    return value;
}
