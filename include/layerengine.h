#ifndef LAYERENGINE_H
#define LAYERENGINE_H

#include "schnode.h"
#include "spatial_mapping/light_placement.h"

class CoreMapper;
//#include "coremapping.h"

struct LayerScheme{
	// Total cost.
	SchNode::SchCost totCost;
	// Used to update external ubuf cost.
	energy_t extUbufEnergy;
	access_t dram_access;
	vol_t weight_buf;
	vol_t ifmap_buf;
	vol_t ofmap_buf;
	CoreMapper::CoreMapping tileSch;
	PlaceSch place;
	NoC noc;
	bool isValid() const;
};


class LayerEngine{
public:
	virtual LayerScheme search(LNode* curNode, bool calc_noc = true) const = 0;
	// TODO: put it somewhere else.
	virtual vol_t get_ubuf_size() const = 0;
	virtual LayerScheme fillin(LNode* curNode, const Light_placement &place, bool calc_noc = true,bool base=false) = 0;
};

class StdLayerEngine : public LayerEngine{
	CoreMapper* mapper;
public:
	StdLayerEngine(CoreMapper* _mapper);
	virtual vol_t get_ubuf_size() const override;
	virtual LayerScheme search(LNode* curNode, bool calc_noc = true) const override;
	void initLayouts(PlaceSch& place, const Node& layerT, const fmap_shape& ofmShape, len_t B) const;
	void initLayouts_by_light(PlaceSch& place, const Node& layerT, const fmap_shape& ofmShape, len_t B, const Light_placement &light_place) const;
	void calcNoC(NoC& noc, const PlaceSch& place, LNode* curNode, bool unordered = false) const;
	void calcNoC(NoC& noc, const PlaceSch& place, const Light_placement& light_place, LNode* curNode, bool unordered = false) const;
	virtual LayerScheme fillin(LNode* curNode, const Light_placement &place, bool calc_noc = true,bool base=false);
};

class FastLayerEngine : public LayerEngine {
	CoreMapper* mapper;
public:
	FastLayerEngine(CoreMapper* _mapper);
	virtual vol_t get_ubuf_size() const override;
	virtual LayerScheme search(LNode* curNode, bool calc_noc = true) const override;
	void initLayouts(PlaceSch& place, const Node& layerT, const fmap_shape& ofmShape, len_t B) const;
	virtual LayerScheme fillin(LNode* curNode, const Light_placement &place, bool calc_noc = true, bool base = false);
	PartSch fast_get_part(Cluster c, const Node& layerT, len_t num_batch);
	
	//void calcNoC(NoC& noc, const PlaceSch& place, LNode* curNode) const;
};

#endif // LAYERENGINE_H
