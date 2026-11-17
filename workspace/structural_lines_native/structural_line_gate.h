#pragma once

#include "structural_lines.h"
#include <opencv2/calib3d.hpp>
#include <limits>

namespace structural_lines {

// Frozen engineering bounds, not calibrated probabilities. No direction is
// interpreted as a known global axis; all constraints are between past images.
struct GateConfig {
    static constexpr int interval = 8;
    static constexpr int decision_phase = 6;
    static constexpr int spread_frames = 2;
    static constexpr double gain = .1;
    static constexpr double maximum_raw_rad = .02;
    static constexpr double maximum_applied_rad = .002;
    static constexpr double minimum_pair_angle_rad = .5235987755982988; // 30 deg
    static constexpr double matching_angle_rad = .08726646259971647;    // 5 deg
    static constexpr double pair_shape_error_rad = .017453292519943295; // 1 deg
    static constexpr double maximum_fit_error_rad = .008726646259971648;// .5 deg
    static constexpr double ambiguity_margin_rad = .004363323129985824; // .25 deg
    static constexpr double temporal_difference_rad = .008726646259971648;
    static constexpr double minimum_information_ratio = .03;
    static constexpr double maximum_frame_gap_seconds = .2;
    static constexpr double maximum_pair_gap_seconds = .6;
};

enum class GateReason : int {
    none=0, accepted=1, weak=2, match=3, ambiguous=4, shape=5,
    residual=6, temporal=7, point_overlap=8, gap=9
};

inline bool finite_matrix(const Matx33d& m) {
    for (double x : m.val) if (!std::isfinite(x)) return false;
    return true;
}
inline double bounded_acos(double x) {
    return std::acos(std::max(-1., std::min(1., x)));
}
inline double direction_angle(const Vec3d& a, const Vec3d& b) {
    return bounded_acos(std::abs(a.dot(b)));
}
inline Matx33d gate_exp(const Vec3d& v) {
    Matx33d r; cv::Rodrigues(v,r); return r;
}
inline Vec3d gate_log(const Matx33d& r) {
    Vec3d v; cv::Rodrigues(r,v); return v;
}
inline bool eligible_direction(const Direction& d) {
    return std::isfinite(d.axis[0]) && std::isfinite(d.axis[1]) &&
        std::isfinite(d.axis[2]) && std::abs(norm3(d.axis)-1.)<1e-5 &&
        d.support>=3 && std::isfinite(d.residual) && d.residual>=0. &&
        d.residual<=.013962634015954637 && std::isfinite(d.conditioning) &&
        d.conditioning>=.004 && d.conditioning<=1. &&
        std::isfinite(d.center_span_px) && d.center_span_px>=18.;
}
inline bool has_two_directions(const Detection& d) {
    if (!d.valid_image) return false;
    for (size_t i=0;i<d.directions.size();++i)
        for (size_t j=i+1;j<d.directions.size();++j)
            if (eligible_direction(d.directions[i]) && eligible_direction(d.directions[j]) &&
                direction_angle(d.directions[i].axis,d.directions[j].axis)>=GateConfig::minimum_pair_angle_rad)
                return true;
    return false;
}

struct RelativeFit {
    bool valid=false;
    GateReason reason=GateReason::weak;
    Matx33d rotation=Matx33d::eye(); // Anchor camera <- current camera.
    double innovation_rad=0., maximum_error_rad=0.;
    int matched_directions=0;
};

// Equal-weight Wahba fit to independently supported, unoriented directions.
// In particular, we never construct an orthogonal room frame from these axes.
inline RelativeFit fit_relative_directions(const Detection& anchor, const Detection& current,
                                           const Matx33d& predicted) {
    RelativeFit result;
    if (!has_two_directions(anchor) || !has_two_directions(current) ||
        !finite_matrix(predicted)) return result;
    struct Assignment {
        std::array<int,3> a{},b{};
        std::array<double,3> sign{};
        int count=0;
        double cost=0.;
    };
    std::vector<Assignment> assignments;
    bool had_correspondence=false;
    auto consider=[&](std::array<int,3> ai,std::array<int,3> bi,int count) {
        Assignment candidate; candidate.a=ai; candidate.b=bi; candidate.count=count;
        for (int j=0;j<count;++j) {
            const auto& a=anchor.directions[ai[j]];
            const auto& b=current.directions[bi[j]];
            if (!eligible_direction(a) || !eligible_direction(b)) return;
            const double dot=a.axis.dot(predicted*b.axis);
            const double angle=bounded_acos(std::abs(dot));
            if (angle>GateConfig::matching_angle_rad) return;
            candidate.sign[j]=dot<0.?-1.:1.;
            candidate.cost+=angle*angle/double(count);
        }
        had_correspondence=true;
        for (int i=0;i<count;++i) for (int j=i+1;j<count;++j) {
            const double a=direction_angle(anchor.directions[ai[i]].axis,anchor.directions[ai[j]].axis);
            const double b=direction_angle(current.directions[bi[i]].axis,current.directions[bi[j]].axis);
            if (std::min(a,b)<GateConfig::minimum_pair_angle_rad ||
                std::abs(a-b)>GateConfig::pair_shape_error_rad) return;
        }
        assignments.push_back(candidate);
    };
    const int na=std::min(3,int(anchor.directions.size()));
    const int nb=std::min(3,int(current.directions.size()));
    if (na==3 && nb==3)
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k)
            if(i!=j && i!=k && j!=k) consider({0,1,2},{i,j,k},3);
    if (assignments.empty())
        for(int a=0;a<na;++a) for(int b=a+1;b<na;++b)
            for(int i=0;i<nb;++i) for(int j=0;j<nb;++j)
                if(i!=j) consider({a,b,0},{i,j,0},2);
    if (assignments.empty()) {
        result.reason=had_correspondence?GateReason::shape:GateReason::match; return result;
    }
    std::sort(assignments.begin(),assignments.end(),[](const Assignment& a,const Assignment& b){return a.cost<b.cost;});
    if (assignments.size()>1 && assignments[1].cost-assignments[0].cost<
        GateConfig::ambiguity_margin_rad*GateConfig::ambiguity_margin_rad) {
        result.reason=GateReason::ambiguous; return result;
    }
    const auto& selected=assignments.front();
    Matx33d cross=Matx33d::zeros();
    for(int j=0;j<selected.count;++j) {
        const auto a=anchor.directions[selected.a[j]].axis;
        const auto b=current.directions[selected.b[j]].axis*selected.sign[j];
        for(int r=0;r<3;++r) for(int c=0;c<3;++c) cross(r,c)+=a[r]*b[c];
    }
    Vec3d singular; Matx33d u,vt;
    cv::SVD::compute(cross,singular,u,vt);
    Matx33d proper=Matx33d::eye();
    proper(2,2)=cv::determinant(u*vt)<0.?-1.:1.;
    result.rotation=u*proper*vt;
    Matx33d information=Matx33d::zeros();
    for(int j=0;j<selected.count;++j) {
        const auto a=anchor.directions[selected.a[j]].axis;
        const auto b=current.directions[selected.b[j]].axis*selected.sign[j];
        const auto mapped=result.rotation*b;
        result.maximum_error_rad=std::max(result.maximum_error_rad,bounded_acos(a.dot(mapped)));
        for(int r=0;r<3;++r) for(int c=0;c<3;++c)
            information(r,c)+=(r==c?1.:0.)-mapped[r]*mapped[c];
    }
    Vec3d eigen;
    if (!cv::eigen(information,eigen) || eigen[2]<
        GateConfig::minimum_information_ratio*(eigen[0]+eigen[1]+eigen[2])) {
        result.reason=GateReason::shape; return result;
    }
    result.innovation_rad=norm3(gate_log(result.rotation*predicted.t()));
    if (!finite_matrix(result.rotation) || !std::isfinite(result.innovation_rad) ||
        result.innovation_rad>GateConfig::maximum_raw_rad ||
        result.maximum_error_rad>GateConfig::maximum_fit_error_rad) {
        result.reason=GateReason::residual; return result;
    }
    result.valid=true; result.reason=GateReason::accepted;
    result.matched_directions=selected.count;
    return result;
}

// A_target maps the unchanged gyro frame into the emitted world frame. The
// sampled raw rotation stays immutable even if processing finishes later.
inline Vec3d alignment_target_delta(const Matx33d& anchor_output,
                                    const Matx33d& relative_visual,
                                    const Matx33d& sampled_raw_rotation,
                                    const Matx33d& current_alignment) {
    const Matx33d target=anchor_output*relative_visual*sampled_raw_rotation.t();
    return gate_log(target*current_alignment.t());
}

class Controller {
public:
    void configure(double fx320,double fy320,double cx320,double cy320) {
        *this=Controller();
        fx_=fx320*.5; fy_=fy320*.5;
        cx_=(cx320+.5)*.5-.5; cy_=(cy320+.5)*.5-.5;
        configured_=std::isfinite(fx_) && std::isfinite(fy_) &&
            std::isfinite(cx_) && std::isfinite(cy_) && fx_>0. && fy_>0.;
    }

    Matx33d step(const uint8_t* rgb,const Matx33d& raw_rotation,
                 const Matx33d& current_alignment,double time,
                 bool point_rotation_active,double point_fits,bool enabled) {
        Matx33d aligned=current_alignment;
        const bool point_event=point_rotation_active ||
            (started_ && point_fits!=last_point_fits_);
        if (point_event) {
            cancel_pending(); anchor_valid_=false; consecutive_=0;
            previous_residual_valid_=false; overlap_=true;
        }
        const bool gap=started_ && (!std::isfinite(time) || time<=previous_time_ ||
            time-previous_time_>GateConfig::maximum_frame_gap_seconds+1e-9);
        if (gap || !finite_matrix(raw_rotation) || !finite_matrix(current_alignment)) {
            cancel_pending(); anchor_valid_=false; captured_=false;
            consecutive_=0; previous_residual_valid_=false; count_reason(GateReason::gap);
        }
        const int phase=frame_index_%GateConfig::interval;
        if (phase==0 && rgb && configured_ && std::isfinite(time)) {
            capture_gray(rgb);
            capture_.raw=raw_rotation;
            capture_.output=current_alignment*raw_rotation;
            capture_.time=time;
            captured_=true;
            ++diag_[0]; diag_[16]=double(frame_index_);
        }
        if (phase==GateConfig::decision_phase && captured_ && configured_) {
            last_detection_=detector_.detect(image_,fx_,fy_,cx_,cy_);
            ++diag_[2]; diag_[17]=double(last_detection_.lines.size());
            diag_[18]=double(last_detection_.directions.size()); diag_[19]=0.;
            const bool usable=has_two_directions(last_detection_);
            if (usable) ++diag_[1];
            decide(usable,aligned,enabled);
            captured_=false;
        }
        if (pending_frames_>0) {
            // Only phases 6 and 7 can apply a fresh scheduled correction.
            if (enabled) {
                aligned=gate_exp(pending_world_)*aligned;
                ++diag_[13]; diag_[15]+=norm3(pending_world_);
            }
            --pending_frames_;
        }
        previous_time_=time; last_point_fits_=point_fits;
        started_=true; ++frame_index_;
        return aligned;
    }

    std::array<double,21> diagnostics() const { return diag_; }
    const Detection& last_detection() const { return last_detection_; }

private:
    friend struct ControllerTestAccess;
    struct Snapshot {
        Matx33d raw=Matx33d::eye(),output=Matx33d::eye();
        double time=0.;
        Detection observation;
    } capture_,anchor_;
    Detector detector_;
    Detection last_detection_;
    cv::Mat image_;
    double fx_=1.,fy_=1.,cx_=0.,cy_=0.,previous_time_=0.,last_point_fits_=0.;
    int frame_index_=0,consecutive_=0,pending_frames_=0;
    bool configured_=false,started_=false,captured_=false,anchor_valid_=false;
    bool overlap_=false,previous_residual_valid_=false;
    Vec3d pending_world_{0.,0.,0.},previous_residual_{0.,0.,0.};
    std::array<double,21> diag_{};

    void count_reason(GateReason reason) {
        const int code=int(reason);
        if(code>=1 && code<=9) ++diag_[code+2];
        diag_[20]=double(code);
    }
    void cancel_pending() {
        if(pending_frames_>0) ++diag_[12];
        pending_frames_=0; pending_world_={0.,0.,0.};
    }
    void capture_gray(const uint8_t* rgb) {
        image_.create(120,160,CV_8UC1);
        for(int y=0;y<120;++y) {
            auto* dest=image_.ptr<uint8_t>(y);
            const auto* src=rgb+(2*y)*320*3+1;
            for(int x=0;x<160;++x,src+=6)
                dest[x]=uint8_t((int(src[0])+int(src[3])+int(src[960])+int(src[963])+2)/4);
        }
    }
    void seed_anchor() {
        anchor_=capture_; anchor_.observation=last_detection_; anchor_valid_=true;
    }
    void decide(bool usable,const Matx33d& aligned,bool enabled) {
        if (overlap_) {
            count_reason(GateReason::point_overlap);
            overlap_=false; anchor_valid_=false; consecutive_=0;
            previous_residual_valid_=false; return;
        }
        if (!usable) {
            count_reason(GateReason::weak); anchor_valid_=false; consecutive_=0;
            previous_residual_valid_=false; return;
        }
        if (!anchor_valid_) {
            seed_anchor(); consecutive_=0; previous_residual_valid_=false;
            count_reason(GateReason::temporal); return;
        }
        const double interval=capture_.time-anchor_.time;
        if (!(interval>0. && interval<=GateConfig::maximum_pair_gap_seconds+1e-9)) {
            count_reason(GateReason::gap); seed_anchor(); consecutive_=0;
            previous_residual_valid_=false; return;
        }
        const Matx33d predicted=anchor_.output.t()*aligned*capture_.raw;
        const auto fit=fit_relative_directions(anchor_.observation,last_detection_,predicted);
        diag_[19]=fit.innovation_rad;
        if (!fit.valid) {
            count_reason(fit.reason); seed_anchor(); consecutive_=0;
            previous_residual_valid_=false; return;
        }
        const Vec3d residual=alignment_target_delta(anchor_.output,fit.rotation,capture_.raw,aligned);
        const double amount=norm3(residual);
        diag_[19]=amount;
        if (!std::isfinite(amount) || amount>GateConfig::maximum_raw_rad) {
            count_reason(GateReason::residual); seed_anchor(); consecutive_=0;
            previous_residual_valid_=false; return;
        }
        const bool stable=previous_residual_valid_ &&
            norm3(residual-previous_residual_)<=GateConfig::temporal_difference_rad;
        previous_residual_=residual; previous_residual_valid_=true;
        consecutive_=stable?consecutive_+1:1;
        seed_anchor();
        if (consecutive_<2) { count_reason(GateReason::temporal); return; }
        Vec3d applied=residual*GateConfig::gain;
        const double applied_amount=norm3(applied);
        if (applied_amount>GateConfig::maximum_applied_rad)
            applied*=GateConfig::maximum_applied_rad/applied_amount;
        count_reason(GateReason::accepted); diag_[14]+=norm3(applied);
        // Detection-only records the counterfactual approval, but never mutates
        // alignment or increments applied-step/angle counters.
        if (enabled && norm3(applied)>1e-12) {
            pending_world_=applied/double(GateConfig::spread_frames);
            pending_frames_=GateConfig::spread_frames;
        }
        // The sampled output at b precedes this delayed update. Reusing b as
        // an emitted-pose anchor would mistake this update for motion in the
        // next interval and partially undo an isolated correction. Reseed only
        // from a new image captured after both scheduled steps have completed.
        anchor_valid_=false; consecutive_=0; previous_residual_valid_=false;
    }
};
} // namespace structural_lines
