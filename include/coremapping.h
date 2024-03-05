#ifndef COREMAPPING_H
#define COREMAPPING_H

#include <cstdint>
#include <string>
#include "core.h"
#include "layer.h"
#include "util.h"

struct PartSch;
//#include "partition.h"

/* Example of for loop:
 *
 * // L3 level (swap these two loops if no L2)
 * for K3
 *   for BH3W3
 *     // L2 level
 *     for K2
 *       // BSD level
 *       for WBSD in 4
 *         for KBSD in 4
 *           // PE level
 *           for K1
 *             for C
 *               for RS
 *                 for H1W1
 *
 * WL2
 * No WBSD and K2>1 -> No KBSD
 * K2=1 -> No WBSD
 * BH3W3=1 or K2=1 -> No L2
 *
 * // L3 level (swap these two loops if no L2)
 * for BH3W3
 *   for K3
 *     // L2 level
 *     for W2
 *       // BSD level
 *       for KBSD in 4
 *         for WBSD in 4
 *           // PE level
 *           for K1
 *             for C
 *               for RS
 *                 for H1W1
 * AL2
 */

/* Ifmap BSD:
 * 1234
 * 2341
 * 3412
 * 4123
 * 5666/5676
 * 6555/6757
 * ----/7565
 *
 */

class CoreMapper{
public:
	typedef ConvLayer::Workload ConvParent;
	struct ConvWl: public ConvParent{
		len_t B, nGroup;
		ConvWl(const ConvParent& parent, len_t _B);
		void init();
		vol_t ifm_size() const;
		vol_t ofm_size() const;
		void calc_op();
	};
	struct MapCost{
		energy_t energy;
		cycle_t time;
		MapCost(energy_t _energy=energy_inf, cycle_t _time=0);
		bool is_valid() const;
		cost_t cost(len_t nbatch=1) const;
	};
	struct CoreMapping{
		MapCost cost;
		energy_t ubuf, buffer, noc, mac;
		double util;
		double tot_util;
		CoreMapping& operator*=(len_t factor);
		CoreMapping& operator+=(const CoreMapping& other);
	};
	// Base core
	const Core& base_core;
	CoreMapper(const Core& c);
	CoreMapping genLayerMap(const Layer& layer, const PartSch& part, const FetchSch& fetch, len_t batch_size, bool wgtB);
	virtual CoreMapping genMapping(const ConvWl& wl) = 0;
	const Core& core() const;
	void set_utime(Layer& l) const;
	virtual void set_conv_utime(ConvLayer& l) const;
	virtual void set_lr_utime(LRLayer& l) const;
	virtual vol_t get_ubuf_size() const = 0;
	virtual ~CoreMapper() = default;
};

class PolarMapper: public CoreMapper{
	class Instance{
	public:
		// Partition...
		enum class PartDim : std::uint8_t {C,K,H,W,B,NPARTS};
		struct Part{
			PartDim dimA, dimO;
			static const char partName[];
			std::string getName() const;
		};
		static const Part all_parts[];
	private:
		// Dataflow...
		enum class DataFlow : std::uint8_t {KHWB,HWBK,NUM};
		struct Loop{
			len_t cnt;
			len_t tot_above;
		};

		// Buffers of current schedule:
		Loop /*TN,*/ H3, /*K2, W2, KBSD, WBSD,*/ K1, H1, W1;
		// Loop K3_KN, K3_NK, W3_KN, W3_NK;
		struct DLoops{
			// For NB, BSD.cnt is invalid
			Loop BSD, L2, L3, L3_N2;
		};
		DLoops K_B,K_NB,W_B,W_NB;

		typedef PolarCore::vmac_t vmac_t;
		//vmac_t part_num[PART_DIM(NPARTS)];
		vmac_t pnum_c,pnum_k,pnum_h,pnum_w,pnum_b;
		vmac_t wdup,adup;
		len_t totTC,totTK,totTH,totTW,totTB;
		len_t totOC,totOK;
		bool hasAL2, hasWL2, hasWBSD, hasKBSD;
		DataFlow curDF;

		vol_t filSize, ofmSize;
		access_t tot_op_k; // Total op using k/VecLen.
		access_t tot_op_c; // Total op using c/VecLen.
		cycle_t tile_time; // Time to compute one tile in a PE.
		cycle_t comp_time;
		vol_t wl1TileSize;

		len_t nDup, TGroup;

		CoreMapping best_map;

		const PolarCore& core;
		// MapCost allcst, hasl2;
		void getCost(const ConvWl& wl);
		//void _try_print(const Workload& wl, size_t i);
	public:
		Instance(const PolarCore& _core);
		CoreMapping genMapping(const ConvWl& wl);
	};
	const PolarCore& core;
public:
	PolarMapper(const PolarCore& _core);
	virtual CoreMapping genMapping(const ConvWl& wl) override;
	virtual void set_conv_utime(ConvLayer& l) const override;
	virtual vol_t get_ubuf_size() const override;
	// virtual ~PolarMapper() override = default;
};

class EyerissMapper : public CoreMapper {
	class Instance{
		/*
		 * Unit_set means mapping a unit workload in a pe stripe.
		 * Pass means mapping multiple unit_set in a PE stripe.
		 * Array means the workload of all the PE array process once.
		 */

		typedef len_t reply_t;
		typedef len_t fold_t;

		// fold_h means fold weight, not necessory in our accelerator.
		// reply_t reply_h, reply_w;
		// fold_t fold_w;
		typedef EyerissCore::vmac_t vmac_t;
		vmac_t phyarr_w;
		reply_t b_reply, k_reply;

		len_t Ct, Kt, Bt, max_c1, max_k1;
		len_t tot_ifmW, tot_ifmH;
		vol_t ifmSize, filSize, ofmSize;
		access_t _al1_buf;

		CoreMapping best_map;

		const EyerissCore& core;
		void getCost(const ConvWl& wl);
	public:
		Instance(const EyerissCore& _core);
		CoreMapping genMapping(const ConvWl& wl);
	};
	const EyerissCore& core;
public:
	EyerissMapper(const EyerissCore& _core);
	virtual CoreMapping genMapping(const ConvWl& wl) override;
	virtual vol_t get_ubuf_size() const override;
	// virtual ~EyerissMapper() override = default;
};
#endif // COREMAPPING_H


