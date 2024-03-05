#ifndef LAYER_H
#define LAYER_H

#include <cstdint>
//#include <iostream>
#include <string>
#include "util.h"
struct PartSch;
struct FetchSch;
//#include "partition.h"

/* With the macros below, one can define layers more conveniently.
 * Example:
 *     a = LAYER("conv1", Conv, C=64, K=128, H=32, R=3)
 * defines a conv layer with a 3*3*64*128 kernel and 32*32*128 ofmap.
 * TYPE of a layer: Conv/FC/LR/Pooling/Eltwise
 */


#ifndef NO_FAST_INIT_LAYER

#define WLarg0_(n, ...)
#define WLarg1_(n, a, ...) n.a;
#define WLarg2_(n, a, b, ...) n.a; n.b;
#define WLarg3_(n, a, b, c, ...) n.a; n.b; n.c;
#define WLarg4_(n, a, b, c, d, ...) n.a; n.b; n.c; n.d;
#define WLarg5_(n, a, b, c, d, e, ...) n.a; n.b; n.c; n.d; n.e;
#define WLarg6_(n, a, b, c, d, e, f, ...) n.a; n.b; n.c; n.d; n.e; n.f;
#define WLarg7_(n, a, b, c, d, e, f, g, ...) n.a; n.b; n.c; n.d; n.e; n.f; n.g;
#define WLarg8_(n, a, b, c, d, e, f, g, h, ...) n.a; n.b; n.c; n.d; n.e; n.f; n.g; n.h;
#define WLarg9_(n, a, b, c, d, e, f, g, h, i, ...) n.a; n.b; n.c; n.d; n.e; n.f; n.g; n.h; n.i;


#define EVAL_Def_(n, a, b, c, d, e, f, g, h, i, m, ...)\
	WLarg##m##_(n, a, b, c, d, e, f, g, h, i)

#define WLDef_(n, ...)\
	EVAL_Def_(n, ##__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define WL_(TYPE, ...)\
	([&]{ TYPE::Workload _x_; WLDef_(_x_, ##__VA_ARGS__) return _x_; }())

#define LAYER(NAME, TYPE, ...)\
	TYPE##Layer(NAME, WL_(TYPE##Layer, ##__VA_ARGS__))

#define NLAYER(NAME, TYPE, ...)\
	new LAYER(NAME, TYPE, ##__VA_ARGS__)

#endif // #ifndef NO_FAST_INIT_LAYER

class Layer{
protected:
	std::string name;
	utime_t unit_time;
	// Ifm_shape contains padding.
	fmap_shape ifm_shape, ofm_shape, wgt_shape;
	bwidth_t bitwidth;
	Layer(const std::string& _name, const fmap_shape& _ifm_shape, const fmap_shape& _ofm_shape, const fmap_shape& _wgt_shape);
	Layer(const std::string& _name);
public:
	bwidth_t get_bitwidth() const;
	void set_bitwidth(bwidth_t width);
	const std::string& get_name() const;
	utime_t get_utime() const;
	void set_utime(utime_t time);
	const fmap_shape& tot_ifmap_shape() const;
	const fmap_shape& ofmap_shape() const;
	const fmap_shape& weight_shape() const;
	virtual FetchSch set_fetch(const PartSch& partSch, vol_t size, len_t B, len_t wgt_B) const = 0;
	virtual vol_t ifm_part(fmap_range& ifm_range, const PartSch& part) const = 0;
	virtual vol_t wgt_part(fmap_range& wgt_range, const PartSch& part) const = 0;

	virtual const fmap_shape& real_ifmap_shape() const =0;
	virtual vol_t weight_size() const =0;
	virtual bool set_padded_ifm(const fmap_shape& padded_shape)=0;
	virtual access_t get_num_op(len_t batch_size=1) const =0;
	virtual void ofm_to_ifm(fmap_range& ofm_range) const =0;
	/* Correspondance between weight(B(1), C, K, R, S) and fmap_range(B,C,H,W):
	 * B = B(1)
	 * C = K
	 * H = C
	 * W = R*S
	 */
	virtual void ofm_to_wgt(fmap_range& ofm_range) const =0;
	virtual bool fmap_channel_rel() const =0;
	virtual ~Layer()=default;
};

class ConvLayer: public Layer{
public:
	struct Workload{
		/* Ifmap: C*[R+(H-1)*sH]*[S+(W-1)*sW]
		 * Filter: C*K*R*S
		 * Ofmap: K*H*W
		 * // B batches in total.
		 * Now there is no batch here.
		 */
		/* Default:
		 * K=C, R=1, S=R, W=H, sH=1, sW=sH
		 */
		// Must init C, H
		len_t C,K=0,R=1,S=0,H,W=0,sH=1,sW=0;
		access_t tot_op;
		void init();
		vol_t ifm_size(len_t batch_size) const;
		vol_t fil_size() const;
		vol_t ofm_size(len_t batch_size) const;
		void update_op();
		access_t calc_op(len_t batch_size) const;
	};
protected:
	Workload wl;
	fmap_shape padded_ifm_shape;
	len_t pad_h, pad_w;
	vol_t wgt_size;
public:
	ConvLayer(const std::string& _name, const Workload& _wl);
	virtual const fmap_shape& real_ifmap_shape() const override;
	virtual vol_t weight_size() const override;
	const Workload& get_workload() const;
	virtual bool set_padded_ifm(const fmap_shape& padded_shape) override;
	virtual access_t get_num_op(len_t batch_size=1) const override;
	virtual void ofm_to_ifm(fmap_range& ofm_range) const override;
	virtual void ofm_to_wgt(fmap_range& ofm_range) const override;
	virtual FetchSch set_fetch(const PartSch& partSch, vol_t size, len_t B, len_t wgt_B) const override;
	virtual vol_t ifm_part(fmap_range& ifm_range, const PartSch& part) const override;
	virtual vol_t wgt_part(fmap_range& wgt_range, const PartSch& part) const override;


	virtual bool fmap_channel_rel() const override;
	virtual ~ConvLayer() override=default;
};

class GroupConvLayer: public ConvLayer{
public:
	struct Workload: ConvLayer::Workload{
		/* Ifmap: (G*C)*[R+(H-1)*sH]*[S+(W-1)*sW]
		 * Filter: G*C*K*R*S
		 * Ofmap: (G*K)*H*W
		 * // B batches in total.
		 * Now there is no batch here.
		 */
		/* Default:
		 * K=C, R=1, S=R, W=H, sH=1, sW=sH
		 */
		// Must init G, C, H
		len_t G;
		// C and K for one group. No need to set.
		len_t GC, GK;
	public:
		void init();
		vol_t fil_size() const;
	};
protected:
	Workload wl;
public:
	GroupConvLayer(const std::string& _name, const Workload& _wl);
	const Workload& get_workload() const;
	virtual void ofm_to_ifm(fmap_range& ofm_range) const override;
	virtual void ofm_to_wgt(fmap_range& ofm_range) const override;
	virtual FetchSch set_fetch(const PartSch& partSch, vol_t size, len_t B, len_t wgt_B) const override;
	virtual vol_t ifm_part(fmap_range& ifm_range, const PartSch& part) const override;
	virtual vol_t wgt_part(fmap_range& wgt_range, const PartSch& part) const override;


	virtual bool fmap_channel_rel() const override;
	virtual ~GroupConvLayer() override=default;
};

class FCLayer: public ConvLayer{
public:
	struct Workload{
		/* Ifmap: C*IH*IW
		 * Filter: C*K*IH*IW
		 * Ofmap: K*1*1
		 * // B batches in total.
		 * Now there is no batch here.
		 */
		len_t C,K,IH=1,IW=0;
	};
	FCLayer(const std::string& _name, const Workload& wl);
	virtual void ofm_to_ifm(fmap_range& ofm_range) const override;
};

// TODO: make LRLayer a non pure-virtual class, handling general cases.
class LRLayer: public Layer{
public:
	struct Workload{
		/* Ifmap: [N+(K-1)*sK]*[R+(H-1)*sH]*[S+(W-1)*sW]
		 * Filter: 0
		 * Ofmap: K*H*W
		 * // B batches in total.
		 * Now there is no batch here.
		 */
		/* Default:
		 * W=H, S=R, sK=N, sH=R, sW=sH
		 * Most of the time we have sK=N, sH=R, sW=sH.
		 */
		// Needs K, H, N, R
		len_t K,H,W=0,N,R,S=0,sK=0,sH=0,sW=0;
		access_t tot_op;
		void init();
		void update_op();
		access_t calc_op(len_t batch_size) const;
	};
protected:
	fmap_shape padded_ifm_shape;
	Workload wl;
	len_t pad_h, pad_w;
	LRLayer(const std::string& _name, const Workload& _wl);
public:
	const Workload& get_workload() const;
	virtual const fmap_shape& real_ifmap_shape() const override;
	virtual vol_t weight_size() const override;
	virtual bool set_padded_ifm(const fmap_shape& padded_shape) override = 0;
	virtual access_t get_num_op(len_t batch_size=1) const override;
	virtual void ofm_to_ifm(fmap_range& ofm_range) const override = 0;
	virtual void ofm_to_wgt(fmap_range& ofm_range) const override;
	virtual FetchSch set_fetch(const PartSch& partSch, vol_t size, len_t B, len_t wgt_B) const override;
	virtual vol_t ifm_part(fmap_range& ifm_range, const PartSch& part) const override;
	virtual vol_t wgt_part(fmap_range& wgt_range, const PartSch& part) const override;


	virtual bool fmap_channel_rel() const override;
	virtual ~LRLayer() override =default;
};

class PoolingLayer: public LRLayer{
public:
	struct Workload{
		/* Ifmap: K*[R+(H-1)*sH]*[S+(W-1)*sW]
		 * Filter: 0
		 * Ofmap: K*H*W
		 * // B batches in total.
		 * Now there is no batch here.
		 */
		/* Default:
		 * W=H, S=R, sH=R, sW=sH
		 * Most of the time we have sH=R, sW=sH.
		 */
		// Needs K, H, R
		len_t K,H,W=0,R,S=0,sH=0,sW=0;
	};
	PoolingLayer(const std::string& _name, const Workload& wl);
	virtual bool set_padded_ifm(const fmap_shape& padded_shape) override;
	virtual void ofm_to_ifm(fmap_range& ofm_range) const override;
	virtual ~PoolingLayer() override =default;
};

class EltwiseLayer: public LRLayer{
public:
	struct Workload{
		/* Ifmap: (N*K)*H*W
		 * Filter: 0
		 * Ofmap: K*H*W
		 * // B batches in total.
		 * Now there is no batch here.
		 */
		len_t N,K,H,W=0;
	};
	EltwiseLayer(const std::string& _name, const Workload& wl);
	virtual bool set_padded_ifm(const fmap_shape& padded_shape) override;
	virtual void ofm_to_ifm(fmap_range& ofm_range) const override;
	virtual ~EltwiseLayer() override =default;
};

class PTPLayer: public LRLayer{
public:
	struct Workload{
		/* Ifmap: K*H*W
		 * Filter: 0
		 * Ofmap: K*H*W
		 * // B batches in total.
		 * Now there is no batch here.
		 */
		// Default: W=H
		// Needs K, H
		len_t K,H,W=0;
	};
	PTPLayer(const std::string& _name, const Workload& wl);
	virtual bool set_padded_ifm(const fmap_shape& padded_shape) override;
	virtual void ofm_to_ifm(fmap_range& ofm_range) const override;
	virtual ~PTPLayer() override =default;
};

class TransposeLayer: public LRLayer{
public:
	enum dim : std::uint8_t{
		C=0,
		H=1,
		W=2,
		NUM=3
	};
	struct Workload{
		/* Ifmap: K*H*W
		 * Filter: 0
		 * Ofmap: K*H*W
		 * // B batches in total.
		 * Now there is no batch here.
		 */
		// Default: W=H
		// Needs K, H
		len_t K,H,W=0;
		dim order[dim::NUM];
		Workload();
		void init();
		fmap_range::dim_range& get_origin_dim(fmap_range& range, dim d) const;
	};
protected:
	Workload wl;
public:
	TransposeLayer(const std::string& _name, const Workload& _wl);
	virtual bool set_padded_ifm(const fmap_shape& padded_shape) override;
	virtual void ofm_to_ifm(fmap_range& ofm_range) const override;
	virtual ~TransposeLayer() override =default;
};

#endif // LAYER_H
