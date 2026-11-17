#pragma once
#include "structural_line_gate.h"
#include <cstdio>
#include <limits>

inline int gate_cases() {
    using namespace structural_lines;
    int failed=0;
    auto check=[&](bool ok,const char* label) {
        std::printf("%s gate: %s\n",ok?"PASS":"FAIL",label); failed+=!ok;
    };
    auto observation=[](const std::vector<Vec3d>& axes,const Matx33d& camera_to_anchor=Matx33d::eye()) {
        Detection d; d.valid_image=true;
        for(auto axis:axes) {
            Direction x; x.axis=camera_to_anchor.t()*canonical(axis);
            x.support=4; x.residual=0.; x.conditioning=.1; x.center_span_px=50.;
            d.directions.push_back(x);
        }
        return d;
    };
    auto error=[](const Matx33d& a,const Matx33d& b) {return norm3(gate_log(a*b.t()));};
    const std::vector<Vec3d> axes{{1.,0.,0.},{.573576436351046,.819152044288992,0.}}; // 55 degrees.
    const Matx33d truth=gate_exp({.21,-.13,.17});
    const auto anchor=observation(axes);
    const auto current=observation(axes,truth);
    for(int axis=0;axis<3;++axis) {
        Vec3d e{0.,0.,0.}; e[axis]=.008;
        const auto predicted=gate_exp(e)*truth;
        const auto fit=fit_relative_directions(anchor,current,predicted);
        check(fit.valid && error(fit.rotation,truth)<1e-7 &&
            std::abs(fit.innovation_rad-.008)<1e-7,
            "nonorthogonal direction pair recovers pure-rotation gyro error on each axis");
    }
    auto exact=fit_relative_directions(anchor,current,truth);
    check(exact.valid && exact.innovation_rad<1e-7,"correct relative pose does not receive a fictitious correction");
    auto altered=current;
    std::swap(altered.directions[0],altered.directions[1]);
    altered.directions[0].axis*=-1.; altered.directions[1].axis*=-1.;
    const auto invariant=fit_relative_directions(anchor,altered,truth);
    check(invariant.valid && error(invariant.rotation,truth)<1e-7,
          "unoriented signs and family permutations preserve relative rotation");

    const auto vertical=observation({{0.,1.,0.}});
    const auto single=fit_relative_directions(vertical,vertical,gate_exp({0.,.01,0.}));
    check(!single.valid && single.reason==GateReason::weak,"one vertical family cannot invent a yaw constraint");
    const auto close=observation({{1.,0.,0.},{1.,.1,0.}});
    check(!fit_relative_directions(close,close,Matx33d::eye()).valid,"nearly parallel families reject rank deficiency");
    const auto wrong_shape=observation({{1.,0.,0.},{.544639035015027,.838670567945424,0.}}); // 57 deg.
    const auto bad_shape=fit_relative_directions(anchor,wrong_shape,Matx33d::eye());
    check(!bad_shape.valid && bad_shape.reason==GateReason::shape,"changing room-axis angle is rejected rather than orthogonalized");
    const auto excessive=fit_relative_directions(anchor,observation(axes,gate_exp({0.,0.,.03})),Matx33d::eye());
    check(!excessive.valid && excessive.reason==GateReason::residual,"oversized innovation is rejected before gain or clipping");
    const auto lost=fit_relative_directions(anchor,observation(axes,gate_exp({0.,0.,.2})),Matx33d::eye());
    check(!lost.valid && lost.reason==GateReason::match,"unmatched cross-room directions reject");
    auto duplicate=anchor; duplicate.directions.push_back(duplicate.directions[0]);
    const auto ambiguous=fit_relative_directions(duplicate,anchor,Matx33d::eye());
    check(!ambiguous.valid && ambiguous.reason==GateReason::ambiguous,"duplicate candidate axes cannot silently choose an ambiguous assignment");
    auto invalid=anchor; invalid.directions[0].residual=.1;
    check(!fit_relative_directions(invalid,current,truth).valid,"invalid family residual cannot become trusted evidence");
    invalid=anchor; invalid.directions[0].axis[1]=std::numeric_limits<double>::quiet_NaN();
    check(!fit_relative_directions(invalid,current,truth).valid,"nonfinite direction rejects");

    const Matx33d ga=gate_exp({.4,.1,-.2}),aa=gate_exp({.1,.2,.3});
    const Matx33d ea=aa*ga,gb=ga*gate_exp({.005,-.004,.003})*truth;
    const Vec3d delta=alignment_target_delta(ea,truth,gb,aa);
    const Matx33d target=ea*truth*gb.t();
    check(error(gate_exp(delta)*aa,target)<1e-7,
          "world-left correction has correct sign and camera-to-world convention");
    check(norm3(alignment_target_delta(ea,truth,gb,target))<1e-7,
          "already-applied V8 alignment is subtracted rather than corrected twice");
    const Matx33d q=gate_exp({-.3,.2,.4});
    const Vec3d rotated=alignment_target_delta(q*ea,truth,gb,q*aa);
    check(norm3(rotated-q*delta)<1e-7,"changing global gauge rotates the correction without inventing a room heading");
    const Matx33d later_g=gb*gate_exp({.1,.05,-.03});
    check(error((gate_exp(delta)*aa)*later_g,target*later_g)<1e-7,
          "delayed update retains captured gyro pose while acting on current pose");

    // Perspective projection of static 3D parallel lines at varied depths.
    // Translation changes line-plane offsets but must not change their VPs.
    auto projected=[](const Vec3d& translation) {
        const double fx=90.,fy=90.,cx=79.5,cy=59.5;
        std::vector<cv::Vec4f> segments;
        const std::vector<Vec3d> directions{{1.,0.,0.},{0.,1.,0.}};
        for(int family=0;family<2;++family) for(int i=0;i<4;++i) {
            const Vec3d center=family==0?Vec3d{0.,-1.5+i,4.+.2*i}:Vec3d{-1.5+i,0.,4.+.2*i};
            auto a=center-directions[family]*1.5-translation;
            auto b=center+directions[family]*1.5-translation;
            segments.push_back({float(fx*a[0]/a[2]+cx),float(fy*a[1]/a[2]+cy),
                                float(fx*b[0]/b[2]+cx),float(fy*b[1]/b[2]+cy)});
        }
        return Detector().from_segments(segments,fx,fy,cx,cy);
    };
    const auto translated=fit_relative_directions(projected({0.,0.,0.}),projected({.2,-.1,.3}),Matx33d::eye());
    check(translated.valid && translated.innovation_rad<1e-6,
          "3D pure translation with perspective and varying depth produces no rotation correction");

    cv::Mat gray(120,160,CV_8UC1,cv::Scalar(20));
    const std::vector<cv::Vec4f> drawn{{5.f,12.f,65.f,12.f},{5.f,48.f,65.f,48.f},{5.f,84.f,65.f,84.f},
        {95.f,8.f,135.f,48.f},{95.f,38.f,135.f,78.f},{95.f,68.f,135.f,108.f}};
    for(const auto& l:drawn) cv::line(gray,{int(l[0]),int(l[1])},{int(l[2]),int(l[3])},cv::Scalar(235),1,cv::LINE_8);
    std::vector<uint8_t> rgb(320*240*3);
    for(int y=0;y<240;++y) for(int x=0;x<320;++x) for(int c=0;c<3;++c)
        rgb[(y*320+x)*3+c]=gray.at<uint8_t>(y/2,x/2);
    Controller detect_only,correcting,repeated,overlap,gapped;
    for(auto* controller:{&detect_only,&correcting,&repeated,&overlap,&gapped}) controller->configure(180.,180.,159.5,119.5);
    Matx33d a=Matx33d::eye(),repeat_a=Matx33d::eye(),overlap_a=Matx33d::eye(),gap_a=Matx33d::eye();
    Matx33d after_isolated=Matx33d::eye();
    bool identical=true,repeat_exact=true,bounds=true;
    for(int frame=0;frame<48;++frame) {
        const double angle=.000625*std::min(frame,16); // No new gyro error after image 16.
        const Matx33d g=gate_exp({0.,0.,angle});
        const auto unchanged=detect_only.step(rgb.data(),g,Matx33d::eye(),frame*.05,false,0.,false);
        for(int i=0;i<9;++i) identical&=unchanged.val[i]==Matx33d::eye().val[i];
        const auto before=a;
        a=correcting.step(rgb.data(),g,a,frame*.05,false,0.,true);
        repeat_a=repeated.step(rgb.data(),g,repeat_a,frame*.05,false,0.,true);
        for(int i=0;i<9;++i) repeat_exact&=a.val[i]==repeat_a.val[i];
        bounds&=error(a,before)<=GateConfig::maximum_applied_rad/GateConfig::spread_frames+1e-7;
        if(frame==23) after_isolated=a;
        overlap_a=overlap.step(rgb.data(),g,overlap_a,frame*.05,frame==23,frame>=23?1.:0.,true);
        gap_a=gapped.step(rgb.data(),g,gap_a,frame*.05+(frame>=23?.3:0.),false,0.,true);
    }
    const auto cd=correcting.diagnostics(),dd=detect_only.diagnostics();
    check(identical && dd[3]>0. && dd[13]==0. && dd[15]==0.,
          "detect-only computes counterfactual approvals while preserving every alignment bit");
    check(cd[3]>0. && cd[13]==2. && cd[15]>0. && cd[15]<=GateConfig::maximum_applied_rad && bounds,
          "real-image temporal controller applies a bounded two-step correction after warmup");
    check(error(a*gate_exp({0.,0.,.01}),Matx33d::eye())<.01 && error(a,after_isolated)<1e-7,
          "isolated accepted correction improves error and a fresh anchor never undoes it");
    check(repeat_exact,"same causal image and IMU prefix gives exact controller repetition");
    check(overlap.diagnostics()[12]==1. && overlap.diagnostics()[10]>0. && overlap.diagnostics()[13]==1.,
          "new V8 rotation cancels the remaining line step and invalidates its anchor");
    check(gapped.diagnostics()[12]==1. && gapped.diagnostics()[11]>0. && gapped.diagnostics()[13]==1.,
          "timestamp gap cancels the remaining step instead of carrying stale evidence");
    std::vector<uint8_t> blank(320*240*3,40);
    Controller weak; weak.configure(180.,180.,159.5,119.5);
    for(int i=0;i<24;++i) weak.step(blank.data(),Matx33d::eye(),Matx33d::eye(),i*.05,false,0.,true);
    check(weak.diagnostics()[4]==3. && weak.diagnostics()[3]==0. && weak.diagnostics()[15]==0.,
          "blank observations reject without scheduling correction");
    return failed;
}
