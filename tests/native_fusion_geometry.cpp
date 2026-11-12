#include "viobench_abi.h"
#include "fusion_geometry.h"
#include <cstdio>

using namespace visual_direction;

static std::vector<Match> scene(const Vec3d& error, bool degenerate=false) {
    std::vector<Match> matches;
    const Vec3d translation{.12,.025,.04};
    const auto r=rotation_matrix(-error);
    for(int i=0;i<48;++i) {
        const double x=degenerate ? .1 : (i%8-3.5)*.18;
        const double y=degenerate ? .1 : (i/8-2.5)*.17;
        const double depth=degenerate ? 3. : 1.5+.37*((i*13)%17);
        const Vec3d a{x,y,1.};
        Vec3d b=r*(a*depth-translation); b*=1./b[2];
        const Vec3d normal=a.cross(b);
        matches.push_back({a,b,normal,normal/std::max(1e-12,length(normal)),.03,1.});
    }
    return matches;
}

extern "C" VIOBENCH_EXPORT(run_tests) int run_tests() {
    int failed=0, tests=0;
    auto check=[&](bool condition,const char* label) {
        ++tests;
        if(!condition) { ++failed; std::printf("FAIL %s\n",label); }
        else std::printf("PASS %s\n",label);
    };
    Vec3d direction{.12,.025,.04}; direction*=1./length(direction);
    const std::array<Vec3d,4> errors{Vec3d(.004,0,0),Vec3d(0,.004,0),
                                  Vec3d(0,0,.004),Vec3d(.003,-.002,.002)};
    for(const auto& error:errors) {
        Vec3d correction;
        const bool accepted=refine_rotation(scene(error),direction,90.,correction);
        std::printf("rotation known %.6f %.6f %.6f, applied %.6f %.6f %.6f\n",
            error[0],error[1],error[2],correction[0],correction[1],correction[2]);
        check(accepted && length(correction-error*.25)<length(error*.25),
              "mixed-depth translation plus gyro error recovers the correct rotation sign");
    }
    Vec3d correction;
    check(!refine_rotation(scene({0.,0.,0.}),direction,90.,correction),
          "translation alone does not create rotation");
    check(!refine_rotation(scene({.004,0.,0.},true),direction,90.,correction),
          "coincident rays reject unobservable rotation");
    auto few=scene({.004,0.,0.}); few.resize(8);
    check(!refine_rotation(few,direction,90.,correction),"too few tracks reject");
    check(!refine_rotation(scene({.004,0.,0.}),direction,0.,correction),"invalid focal rejects");
    auto clean=scene({0.,0.,0.}), bad=clean;
    for(auto& m:bad) { m.backward_error=.6; m.photometric_error=20.; }
    const double goodq=fusion_quality(clean,direction,90.,90.);
    const double badq=fusion_quality(bad,direction,90.,90.);
    check(goodq>badq && badq>=0. && goodq<=1.,"tracking degradation lowers bounded quality");
    check(fusion_quality({},direction,90.,90.)==0.,"empty quality is zero");
    auto outliers=scene({.003,-.002,.002});
    for(size_t i=0;i<outliers.size();i+=4) outliers[i].current={.7,-.5,1.};
    const bool accepted=refine_rotation(outliers,direction,90.,correction);
    check(!accepted || length(correction)<=.00375,"outliers cannot create an unbounded correction");
    std::printf("%d geometry checks, %d failed\n",tests,failed);
    std::fflush(stdout);
    return failed;
}
