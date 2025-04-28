#include "layerengine.h"

#include <cassert>
#include <cstring>

#include "partition.h"
#include "placement.h"


StdLayerEngine::StdLayerEngine(CoreMapper* _mapper):mapper(_mapper){}

FastLayerEngine::FastLayerEngine(CoreMapper* _mapper) :mapper(_mapper) {}

vol_t StdLayerEngine::get_ubuf_size() const{
	return mapper->get_ubuf_size();
}

vol_t FastLayerEngine::get_ubuf_size() const {
	return mapper->get_ubuf_size();
}

LayerScheme FastLayerEngine::fillin(LNode* curNode, const Light_placement &place,bool calc_noc,bool base){

}

LayerScheme StdLayerEngine::fillin(LNode* curNode, const Light_placement &place,bool calc_noc, bool base){
	LayerScheme layerSch;
	const Cluster& cluster = curNode->cluster;
	const Node& layerT = curNode->layert;
	const Layer& layer = layerT.layer();
	const fmap_shape& ofmShape = layer.ofmap_shape();
	len_t wgtBatch = curNode->wgt_bgrp;
	len_t B = curNode->num_batch;
	bool wgt_B = layerT.hasWgtPrevs();
	/*
	len_t K = ofmShape.c;
	len_t H = ofmShape.h;
	len_t W = ofmShape.w;
	*/
	cidx_t numCores = cluster.num_cores();
	const Core::Buffer& ubuf = mapper->core().ubuf();
	vol_t totUbufSize = ubuf.Size * numCores;
	layerSch.noc.calc_bw = NoC::calc_noc_control;
	layerSch.noc.is_base = NoC::interleave;
	// Current scheme
	SchNode::SchCost curCost;
	PlaceSch& placeSch = layerSch.place;
	placeSch.fetch = place.get_fetch_scheme().at(curNode->getLayer().getid());
	//NoC noc(false);
	pos_t* permOrder = new pos_t[numCores];
	placeSch.permuteOrder.reset(permOrder);
	placeSch.ifmLayout = std::make_unique<StdDataLayout>(numCores, permOrder);
	if(layer.weight_size() > 0)
		placeSch.wgtLayout = std::make_unique<StdDataLayout>(numCores, permOrder);
	else
		placeSch.wgtLayout = std::make_unique<StdDataLayout>(0, nullptr);
	placeSch.ofmLayout = std::make_unique<StdULayout>(numCores, permOrder);

	PartSch& partSch = placeSch.part;
	CoreMapper::CoreMapping tileSch;

	// For ubuf energy
	energy_t ubufOfm = ofmShape.tot_size(B) * ubuf.RCost;
	energy_t ubufTotal;

	len_t minCuts = 0;
	if(REF_IS_INSTANCE(layer, ConvLayer) && !REF_IS_INSTANCE(layer, GroupConvLayer))
		minCuts = static_cast<len_t>(layer.real_ifmap_shape().tot_size(B) / (totUbufSize*0.8) + 1);
	
	Light_partition partition;
	for(auto layer : place.get_partition()){
		if(layer.layerno == curNode->getLayer().getid()){
			partSch.K = layer.c;
			partSch.B = layer.b;
			partSch.W = layer.w;
			partSch.H = layer.h;
			partition = layer;
		}
	}
	/*
	for(auto core : place.get_placement()){
		if(curNode->getLayer().getid() == core.layerno){
			auto coord = range_from_partition_number(ofmShape, B, partition, core.partno); 
		}
	}*/

		assert(partSch.size() == static_cast<unsigned>(numCores));

		// Update placeSch
		initLayouts_by_light(placeSch, layerT, ofmShape, B, place);

		// Estimate buffer usage
		vol_t estimatedBuf = ofm_ubuf_vol;
		estimatedBuf += placeSch.ifmLayout->maxRange();
		estimatedBuf += placeSch.wgtLayout->maxRange();
		
		// Search for intra-tile dataflow
		tileSch = mapper->genLayerMap(layer, partSch, placeSch.fetch, B, wgt_B);
		if(!tileSch.cost.is_valid()){
			curNode->valid=false;
			return layerSch;
		}
		curCost.energy = tileSch.cost.energy * numCores;
		curCost.time = tileSch.cost.time;

		// Calc ubuf energy
		// TODO: default to not pinning weights.
		energy_t ubufWgt = placeSch.wgtLayout->totalSize() * ubuf.WCost;
		if(!wgt_B) ubufWgt /= (wgtBatch / B);
		ubufTotal = ubufWgt * placeSch.fetch.wgtFetch + ubufOfm;
		ubufTotal += placeSch.ifmLayout->totalSize() * ubuf.WCost * placeSch.fetch.ifmFetch;
		curCost.energy += ubufTotal;

		// Search for placement.
		//auto placeIter = placeEngine.init(placeSch);
		// if(!placeIter) continue;
		{/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			placeSch.initPlacement_by_light(cluster, place, curNode->getLayer().getid());			
			static_cast<StdDataLayout&>(placeSch.getIfmL()).setCPosArr();
			static_cast<StdDataLayout&>(placeSch.getWgtL()).setCPosArr();
			placeSch.finalize();
			calcNoC(layerSch.noc, placeSch, place, curNode, true);
			SchNode::SchCost curCostAll = curCost;
			curCostAll.energy += layerSch.noc.get_cost();
			cycle_t nocTime = layerSch.noc.get_time();
			curCostAll.time = MAX(curCostAll.time, nocTime);
			layerSch.totCost = curCostAll;
			layerSch.extUbufEnergy = ubufTotal;
			layerSch.tileSch = tileSch;
				//layerSch.place.update(std::move(placeSch));
			//}
		}
	
// 	if(layerSch.isValid()){
// 		layerSch.place.ifmLayout = std::move(placeSch.ifmLayout);
// 		layerSch.place.wgtLayout = std::move(placeSch.wgtLayout);
// 		layerSch.place.ofmLayout = std::move(placeSch.ofmLayout);
// 		layerSch.place.permuteOrder = std::move(placeSch.permuteOrder);
// 		initLayouts_by_light(layerSch.place, layerT, ofmShape, B, place);
// 		layerSch.place.initPlacement_by_light(cluster, place);
// 		static_cast<StdDataLayout&>(layerSch.place.getIfmL()).setCPosArr();
// 		static_cast<StdDataLayout&>(layerSch.place.getWgtL()).setCPosArr();
// 		layerSch.place.finalize();
// 		calcNoC(layerSch.noc, layerSch.place, curNode);
// 	}
	return layerSch;
}


/**
 * @brief StdLayerEngine::search.
 * Searches partition and placement of each layer.
 * The procedure is as follows
 *  for each partition:
 *      estimate max ubuf usage
 *      search for intra-tile dataflow
 *      calculate ubuf energy (outside tile)
 *      for each placement:
 *		    calculate NoC
 *          update best scheme
 *
 * @return LayerScheme.
 */
LayerScheme StdLayerEngine::search(LNode* curNode,bool calc_noc) const{
	// The final scheme
	LayerScheme layerSch;

	// Info extracted from curNode
	const Cluster& cluster = curNode->cluster;
	const Node& layerT = curNode->layert;
	const Layer& layer = layerT.layer();
	const fmap_shape& ofmShape = layer.ofmap_shape();
	len_t wgtBatch = curNode->wgt_bgrp;
	len_t B = curNode->num_batch;
	bool wgt_B = layerT.hasWgtPrevs();
	/*
	len_t K = ofmShape.c;
	len_t H = ofmShape.h;
	len_t W = ofmShape.w;
	*/
	cidx_t numCores = cluster.num_cores();
	const Core::Buffer& ubuf = mapper->core().ubuf();
	vol_t totUbufSize = ubuf.Size * numCores;

	// Current scheme
	SchNode::SchCost curCost;
	PlaceSch placeSch;
	NoC noc(false, NoC::interleave);
	pos_t* permOrder = new pos_t[numCores];
	placeSch.permuteOrder.reset(permOrder);
	placeSch.ifmLayout = std::make_unique<StdDataLayout>(numCores, permOrder);
	if(layer.weight_size() > 0)
		placeSch.wgtLayout = std::make_unique<StdDataLayout>(numCores, permOrder);
	else
		placeSch.wgtLayout = std::make_unique<StdDataLayout>(0, nullptr);
	placeSch.ofmLayout = std::make_unique<StdULayout>(numCores, permOrder);
	// TODO: change interleaving into 1-2 remote mems.
	/*
	if(curNode->to_dram){
		// TODO: change interleaving into 1-2 remote mems.
		fmap_range ofmRange(ofmShape, totBatch);
		placeSch.memLayout = std::make_unique<MemULayout>(ofmRange, NoC::dram_list.data(), NoC::dram_list.size());
		layerSch.place.memLayout = std::make_unique<MemULayout>(ofmRange, NoC::dram_list.data(), NoC::dram_list.size());
	}
	*/
	PartSch& partSch = placeSch.part;

	CoreMapper::CoreMapping tileSch;

	// For ubuf energy
	energy_t ubufOfm = ofmShape.tot_size(B) * ubuf.RCost;
	energy_t ubufTotal;

	len_t minCuts = 0;
	if(REF_IS_INSTANCE(layer, ConvLayer) && !REF_IS_INSTANCE(layer, GroupConvLayer))
		minCuts = static_cast<len_t>(layer.real_ifmap_shape().tot_size(B) / (totUbufSize*0.8) + 1);
	auto partIter = partEngine.init(numCores, B, layerT, partSch, minCuts);
	if(!partIter){
		return layerSch;
	}
	do{
		assert(partSch.size() == static_cast<unsigned>(numCores));

		// Update placeSch
		initLayouts(placeSch, layerT, ofmShape, B);

		// Estimate buffer usage
		vol_t estimatedBuf = ofm_ubuf_vol;
		estimatedBuf += placeSch.ifmLayout->maxRange();
		estimatedBuf += placeSch.wgtLayout->maxRange();
		if (estimatedBuf > ubuf.Size) {
			if (!curNode->is_seg()) {
				continue;
			}
			placeSch.fetch = layer.set_fetch(partSch, ubuf.Size - ofm_ubuf_vol, B, wgt_B);
			if (!placeSch.fetch) {
				// No fetch scheme available.
				continue;
			}
			/*
			fmap_range r_weight;
			vol_t weight = 0;
			if (placeSch.wgtLayout->maxRange() != 0) {
				r_weight = placeSch.wgtLayout->MaxRange_();
				weight = layer.wgt_part(r_weight, placeSch.fetch);
			}
			fmap_range r_ifmap = placeSch.ifmLayout->MaxRange_();
			vol_t ifmap = layer.ifm_part(r_ifmap, placeSch.fetch);
			if (ifmap * 2 + weight + ofm_ubuf_vol > ubuf.Size) {
				continue;
			}*/
		}
		else {
			// No fetch.
			placeSch.fetch.clear();
		}
		
		// Search for intra-tile dataflow
		tileSch = mapper->genLayerMap(layer, partSch, placeSch.fetch, B, wgt_B);
		if(!tileSch.cost.is_valid()) continue;
		curCost.energy = tileSch.cost.energy * numCores;
		curCost.time = tileSch.cost.time;

		// Calc ubuf energy
		// TODO: default to not pinning weights.
		energy_t ubufWgt = placeSch.wgtLayout->totalSize() * ubuf.WCost;
		if(!wgt_B) ubufWgt /= (wgtBatch / B);
		ubufTotal = ubufWgt * placeSch.fetch.wgtFetch + ubufOfm;
		ubufTotal += placeSch.ifmLayout->totalSize() * ubuf.WCost * placeSch.fetch.ifmFetch;
		curCost.energy += ubufTotal;

		// Search for placement.
		auto placeIter = placeEngine.init(placeSch);
		// if(!placeIter) continue;
		do{
			placeSch.initPlacement(cluster);
			static_cast<StdDataLayout&>(placeSch.getIfmL()).setCPosArr();
			static_cast<StdDataLayout&>(placeSch.getWgtL()).setCPosArr();

			calcNoC(noc, placeSch, curNode);
			SchNode::SchCost curCostAll = curCost;
			curCostAll.energy += noc.get_cost();
			cycle_t nocTime = noc.get_dram_time();
			curCostAll.time = MAX(curCostAll.time, nocTime);
			if(curCostAll.cost() < layerSch.totCost.cost()){
				layerSch.totCost = curCostAll;
				layerSch.extUbufEnergy = ubufTotal;
				layerSch.tileSch = tileSch;
				layerSch.place.update(std::move(placeSch));
			}
		// TODO: add cur_cost.cost back.
		}while(placeIter.nextPlace(/*cur_cost.cost()*/));
	}while(partIter.nextPart(/*cur_cost.cost()*/));
	if(layerSch.isValid()){
		layerSch.place.ifmLayout = std::move(placeSch.ifmLayout);
		layerSch.place.wgtLayout = std::move(placeSch.wgtLayout);
		layerSch.place.ofmLayout = std::move(placeSch.ofmLayout);
		layerSch.place.permuteOrder = std::move(placeSch.permuteOrder);
		initLayouts(layerSch.place, layerT, ofmShape, B);
		layerSch.place.initPlacement(cluster);
		static_cast<StdDataLayout&>(layerSch.place.getIfmL()).setCPosArr();
		static_cast<StdDataLayout&>(layerSch.place.getWgtL()).setCPosArr();
		layerSch.place.finalize();
		layerSch.noc.calc_bw = NoC::calc_noc_control;
		layerSch.noc.is_base = NoC::interleave;
		calcNoC(layerSch.noc, layerSch.place, curNode);
	}
	return layerSch;
}

/**
 * @brief FastLayerEngine::search.
 * Evaluate DDR accesses and buffer usage of each layer.
 * The procedure is as follows

 *
 * @return LayerScheme.
 */
LayerScheme FastLayerEngine::search(LNode* curNode,bool calc_noc) const {
	// The final scheme
	LayerScheme layerSch;
	layerSch.dram_access = 0;
	// Info extracted from curNode
	const Cluster& cluster = curNode->cluster;
	const Node& layerT = curNode->layert;
	const Layer& layer = layerT.layer();
	const fmap_shape& ofmShape = layer.ofmap_shape();
	len_t wgtBatch = curNode->wgt_bgrp;
	len_t B = curNode->num_batch;
	bool wgt_B = layerT.hasWgtPrevs();
	/*
	len_t K = ofmShape.c;
	len_t H = ofmShape.h;
	len_t W = ofmShape.w;
	*/
	cidx_t numCores = cluster.num_cores();
	const Core::Buffer& ubuf = mapper->core().ubuf();
	vol_t totUbufSize = ubuf.Size * numCores;
	if (wgt_B) {
		// Fetch each prev layer from its ofmap/mem layout
		const auto& prevs = layerT.getWgtPrevs();
		FOR_BITSET(it, prevs) {
			lid_t prev = it;
			if (!curNode->get_dirp_set().contains(prev)) {
				layerSch.dram_access += network->getNode(it).layer().weight_size() * B;
			}
		}
	}
	else {
		layerSch.dram_access += layerT.layer().weight_size()/(curNode->wgt_bgrp/B);
	}
	len_t curC = layerT.get_external_C();
	if (curC != 0) {
		layerSch.dram_access+=curC/ layerT.layer().real_ifmap_shape().c * layerT.layer().tot_ifmap_shape().tot_size(B);
	}
	const auto& prevs = layerT.getIfmPrevs();
	FOR_BITSET(it, prevs) {
		lid_t prev = it;
		len_t prevC = network->getNode(prev).layer().ofmap_shape().c;
		if (!curNode->get_dirp_set().contains(prev)) {
			layerSch.dram_access += network->getNode(prev).layer().ofmap_shape().tot_size(B);
		}
	}
	if(curNode->to_dram){
		layerSch.dram_access += layerT.layer().ofmap_shape().tot_size(B);
	}
	layerSch.totCost.time = layerT.layer().get_utime() * B;
	//evaluate buffer
	SchNode::SchCost curCost;
	PlaceSch placeSch;
	//NoC noc(false);
	pos_t* permOrder = new pos_t[numCores];
	placeSch.permuteOrder.reset(permOrder);
	placeSch.ifmLayout = std::make_unique<StdDataLayout>(numCores, permOrder);
	if (layer.weight_size() > 0)
		placeSch.wgtLayout = std::make_unique<StdDataLayout>(numCores, permOrder);
	else
		placeSch.wgtLayout = std::make_unique<StdDataLayout>(0, nullptr);
	placeSch.ofmLayout = std::make_unique<StdULayout>(numCores, permOrder);
	// TODO: change interleaving into 1-2 remote mems.
	/*
	if(curNode->to_dram){
		// TODO: change interleaving into 1-2 remote mems.
		fmap_range ofmRange(ofmShape, totBatch);
		placeSch.memLayout = std::make_unique<MemULayout>(ofmRange, NoC::dram_list.data(), NoC::dram_list.size());
		layerSch.place.memLayout = std::make_unique<MemULayout>(ofmRange, NoC::dram_list.data(), NoC::dram_list.size());
	}
	*/
	PartSch& partSch = placeSch.part;
	//CoreMapper::CoreMapping tileSch;

	// For ubuf energy
	//energy_t ubufOfm = ofmShape.tot_size(B) * ubuf.RCost;
	//energy_t ubufTotal;

	len_t minCuts = 0;
	if (REF_IS_INSTANCE(layer, ConvLayer) && !REF_IS_INSTANCE(layer, GroupConvLayer))
		minCuts = static_cast<len_t>(layer.real_ifmap_shape().tot_size(B) / (totUbufSize * 0.8) + 1);
	auto partIter = partEngine.init(numCores, B, layerT, partSch, minCuts);
	if (!partIter) {
		return layerSch;
	}
	vol_t total_buffer = 0;
	vol_t weight_buf = 0;
	vol_t ifmap_buf = 0;
	vol_t ofmap_buf = 0;
	fmap_range ifmap_range;
	fmap_range weight_range;
	vol_t min_buf = ubuf.Size+1;
	do {
		assert(partSch.size() == static_cast<unsigned>(numCores));

		// Update placeSch
		initLayouts(placeSch, layerT, ofmShape, B);

		// Estimate buffer usage
		ofmap_buf = ofm_ubuf_vol;
		ifmap_buf = placeSch.ifmLayout->maxRange();
		ifmap_range = placeSch.ifmLayout->MaxRange_();
		weight_buf = placeSch.wgtLayout->maxRange();
		weight_range = placeSch.wgtLayout->MaxRange_();
		total_buffer = ofmap_buf + ifmap_buf + weight_buf;
		if (total_buffer > ubuf.Size) {
			if (!curNode->is_seg()) {
				continue;
			}
			placeSch.fetch = layer.set_fetch(partSch, ubuf.Size - ofm_ubuf_vol, B, wgt_B);
			if (!placeSch.fetch) {
				// No fetch scheme available.
				continue;
			}
			ifmap_buf = layer.ifm_part(ifmap_range, placeSch.fetch);
			if (layer.weight_size() > 0) {
				weight_buf = layer.wgt_part(weight_range, placeSch.fetch);
			}
			else {
				weight_buf = 0;
			}
			total_buffer = ofmap_buf + ifmap_buf + weight_buf;
		}
		else {
			// No fetch.
			placeSch.fetch.clear();
		}
		if(total_buffer < min_buf){
			layerSch.totCost.energy = 0;
			layerSch.ifmap_buf = ifmap_buf;
			layerSch.weight_buf = weight_buf;
			layerSch.ofmap_buf = ofmap_buf;
			min_buf = total_buffer;
			//std::cout << "1";
		}
	} while (partIter.nextPart(/*cur_cost.cost()*/));
		// Search for intra-tile dataflow
		//tileSch = mapper->genLayerMap(layer, partSch, B, wgt_B);
		//if (!tileSch.cost.is_valid()) continue;
		//curCost.energy = tileSch.cost.energy * numCores;
		//curCost.time = tileSch.cost.time;

		// Calc ubuf energy
		// TODO: default to not pinning weights.
		//energy_t ubufWgt = placeSch.wgtLayout->totalSize() * ubuf.WCost;
		//if (!wgt_B) ubufWgt /= (totBatch / B);
		//ubufTotal = ubufWgt + ubufOfm;
		//ubufTotal += placeSch.ifmLayout->totalSize() * ubuf.WCost;
		//curCost.energy += ubufTotal;

		// Search for placement.
		//auto placeIter = placeEngine.init(placeSch);
		// if(!placeIter) continue;
		/*
		do {
			placeSch.initPlacement(cluster);
			static_cast<StdDataLayout&>(placeSch.getIfmL()).setCPosArr();
			static_cast<StdDataLayout&>(placeSch.getWgtL()).setCPosArr();
			//calcNoC(noc, placeSch, curNode);
			SchNode::SchCost curCostAll = curCost;
			curCostAll.energy += noc.get_cost();
			cycle_t nocTime = noc.get_time();
			curCostAll.time = MAX(curCostAll.time, nocTime);
			if (curCostAll.cost() < layerSch.totCost.cost()) {
				layerSch.totCost = curCostAll;
				layerSch.extUbufEnergy = ubufTotal;
				layerSch.tileSch = tileSch;
				layerSch.place.update(std::move(placeSch));
			}
			// TODO: add cur_cost.cost back.
		} while (placeIter.nextPlace(/*cur_cost.cost()));
	}
	if (layerSch.isValid()) {
		layerSch.place.ifmLayout = std::move(placeSch.ifmLayout);
		layerSch.place.wgtLayout = std::move(placeSch.wgtLayout);
		layerSch.place.ofmLayout = std::move(placeSch.ofmLayout);
		layerSch.place.permuteOrder = std::move(placeSch.permuteOrder);
		initLayouts(layerSch.place, layerT, ofmShape, B);
		layerSch.place.initPlacement(cluster);
		static_cast<StdDataLayout&>(layerSch.place.getIfmL()).setCPosArr();
		static_cast<StdDataLayout&>(layerSch.place.getWgtL()).setCPosArr();
		layerSch.place.finalize();
		calcNoC(layerSch.noc, layerSch.place, curNode);
	}*/ 
	return layerSch;
}

PartSch FastLayerEngine::fast_get_part(Cluster c, const Node& _layerT, len_t num_batch)  {
	// The final scheme
	LayerScheme layerSch;
	//layerSch.dram_access = 0;
	// Info extracted from curNode
	const Cluster& cluster = c;
	const Node& layerT = _layerT;
	const Layer& layer = layerT.layer();
	const fmap_shape& ofmShape = layer.ofmap_shape();
	//len_t totBatch = LNode::tot_batch;
	len_t B = num_batch;

	cidx_t numCores = cluster.num_cores();
	const Core::Buffer& ubuf = mapper->core().ubuf();
	vol_t totUbufSize = ubuf.Size * numCores;

	layerSch.totCost.time = layerT.layer().get_utime() * B;
	PlaceSch placeSch;
	//NoC noc(false);
	pos_t* permOrder = new pos_t[numCores];
	placeSch.permuteOrder.reset(permOrder);
	placeSch.ifmLayout = std::make_unique<StdDataLayout>(numCores, permOrder);
	if (layer.weight_size() > 0)
		placeSch.wgtLayout = std::make_unique<StdDataLayout>(numCores, permOrder);
	else
		placeSch.wgtLayout = std::make_unique<StdDataLayout>(0, nullptr);
	placeSch.ofmLayout = std::make_unique<StdULayout>(numCores, permOrder);
	PartSch& partSch = placeSch.part;
	PartSch best_part;
	len_t minCuts = 0;
	if (REF_IS_INSTANCE(layer, ConvLayer) && !REF_IS_INSTANCE(layer, GroupConvLayer))
		minCuts = static_cast<len_t>(layer.real_ifmap_shape().tot_size(B) / (totUbufSize * 0.8) + 1);
	auto partIter = partEngine.init(numCores, B, layerT, partSch, minCuts);
	if (!partIter) {
		best_part.B = 0;
		best_part.K = 0;
		best_part.H = 0;
		best_part.W = 0;
		return best_part;
	}
	best_part = partSch;
	vol_t total_buffer = 0;
	vol_t weight_buf = 0;
	vol_t ifmap_buf = 0;
	vol_t ofmap_buf = 0;
	vol_t min_buf = ubuf.Size+1;
	do {
		assert(partSch.size() == static_cast<unsigned>(numCores));

		// Update placeSch
		initLayouts(placeSch, layerT, ofmShape, B);

		// Estimate buffer usage
		ofmap_buf = ofm_ubuf_vol;
		ifmap_buf = placeSch.ifmLayout->maxRange();
		weight_buf = placeSch.wgtLayout->maxRange();
		total_buffer = ofmap_buf + ifmap_buf + weight_buf;
		if (total_buffer > ubuf.Size) continue;
		else if (total_buffer < min_buf) {
			layerSch.ifmap_buf = ifmap_buf;
			layerSch.weight_buf = weight_buf;
			layerSch.ofmap_buf = ofmap_buf;
			min_buf = ifmap_buf + weight_buf + ofmap_buf;
			best_part = placeSch.part;
			//std::cout << "1";
		}
	} while (partIter.nextPart(/*cur_cost.cost()*/));
	if (min_buf == ubuf.Size) {
		best_part.B = 0;
		best_part.K = 0;
		best_part.H = 0;
		best_part.W = 0;
		return best_part;
	}
	return best_part;
}

void StdLayerEngine::initLayouts_by_light(PlaceSch& place, const Node& layerT, const fmap_shape& ofmShape, len_t B, const Light_placement &light_place) const{
	using plen_t = PartSch::partlen_t;

	const PartSch& part = place.part;
	const Layer& layer = layerT.layer();
	bool fmap_K = layer.fmap_channel_rel();
	bool hasWgt = layer.weight_size() > 0;
	bool wgt_B = layerT.hasWgtPrevs();

	fmap_range::dim_range kRange, bRange, hRange, wRange;
	fmap_range emptyRange({0, 0}, {0, 0}, {0, 0}, {0, 0});

	auto& ofmLayout = static_cast<StdULayout&>(place.getOfmL());
	auto& ifmLayout = static_cast<StdDataLayout&>(place.getIfmL());
	auto& wgtLayout = static_cast<StdDataLayout&>(place.getWgtL());
	ofmLayout.setDims(part.K, part.B, part.H, part.W);
	if(fmap_K)
		ifmLayout.setBcast(1, 1);
	else
		ifmLayout.setBcast(part.K, part.B * part.H * part.W);

	if(hasWgt){
		if(wgt_B)
			wgtLayout.setBcast(part.H * part.W, 1);
		else
			wgtLayout.setBcast(part.B * part.H * part.W, 1);
	}
	auto* ofmArr = ofmLayout.rangeArr.get();
	auto* ifmArr = ifmLayout.rangeArr.get();
	auto* wgtArr = wgtLayout.rangeArr.get();

	cidx_t coreid=0;
	for(auto core: light_place.get_placement()){
		if(core.layerno==layerT.getid()){
			*ofmArr=range_from_partition_number(ofmShape, B, part, core.partno);
			vol_t s = (*ofmArr).size();
			ofmLayout.update(*ofmArr);
			cidx_t id=core.partno;
			len_t w = id%part.W;
			id /= part.W;
			len_t h = id%part.H;
			id /= part.H;
			len_t k = id%part.K;
			id /= part.K;
			len_t b = id;
			if(fmap_K || k == 0){
				if(s == 0){
					*ifmArr = emptyRange;
				}else{
					*ifmArr = *ofmArr;
					layer.ofm_to_ifm(*ifmArr);
					ifmLayout.update(*ifmArr);

				}
				++ifmArr;
			}
			if(hasWgt && (wgt_B || b == 0) && h == 0 && w == 0){
				if(s == 0){
					*wgtArr = emptyRange;
				}else{
					*wgtArr = *ofmArr;
					layer.ofm_to_wgt(*wgtArr);
					if(!wgt_B) wgtArr->b = {0, 1};
					wgtLayout.update(*wgtArr);
				}
				++wgtArr;
			}
			++ofmArr;
		}
		++coreid;
	}
	assert(ifmArr == ifmLayout.rangeArr.get() + ifmLayout.rangeLength());
	assert(wgtArr == wgtLayout.rangeArr.get() + wgtLayout.rangeLength());
	assert(ofmArr == ofmLayout.rangeArr.get() + ofmLayout.rangeLength());

	// Multiply size of ifmLayout for eltwise
	auto* eltLayer = dynamic_cast<const EltwiseLayer*>(&layerT.layer());
	if(eltLayer != nullptr){
		ifmLayout.sizeMult(eltLayer->get_workload().N);
	}
}

void StdLayerEngine::initLayouts(PlaceSch& place, const Node& layerT, const fmap_shape& ofmShape, len_t B) const{
	using plen_t = PartSch::partlen_t;

	const PartSch& part = place.part;
	const Layer& layer = layerT.layer();
	bool fmap_K = layer.fmap_channel_rel();
	bool hasWgt = layer.weight_size() > 0;
	bool wgt_B = layerT.hasWgtPrevs();

	len_t* arrs[4];
	arrs[0] = part_intv(ofmShape.c, part.K);
	arrs[1] = part_intv(B, part.B);
	arrs[2] = part_intv(ofmShape.h, part.H);
	arrs[3] = part_intv(ofmShape.w, part.W);
	fmap_range::dim_range kRange, bRange, hRange, wRange;
	fmap_range emptyRange({0, 0}, {0, 0}, {0, 0}, {0, 0});

	auto& ofmLayout = static_cast<StdULayout&>(place.getOfmL());
	auto& ifmLayout = static_cast<StdDataLayout&>(place.getIfmL());
	auto& wgtLayout = static_cast<StdDataLayout&>(place.getWgtL());
	ofmLayout.setDims(part.K, part.B, part.H, part.W);
	if(fmap_K)
		ifmLayout.setBcast(1, 1);
	else
		ifmLayout.setBcast(part.K, part.B * part.H * part.W);

	if(hasWgt){
		if(wgt_B)
			wgtLayout.setBcast(part.H * part.W, 1);
		else
			wgtLayout.setBcast(part.B * part.H * part.W, 1);
	}
	auto* ofmArr = ofmLayout.rangeArr.get();
	auto* ifmArr = ifmLayout.rangeArr.get();
	auto* wgtArr = wgtLayout.rangeArr.get();
	for(plen_t k = 0; k < part.K; ++k){
		kRange = {arrs[0][k], arrs[0][k+1]};
		for(plen_t b = 0; b < part.B; ++b){
			bRange = {arrs[1][b], arrs[1][b+1]};
			for(plen_t h = 0; h < part.H; ++h){
				hRange = {arrs[2][h], arrs[2][h+1]};
				for(plen_t w = 0; w < part.W; ++w){
					wRange = {arrs[3][w], arrs[3][w+1]};
					*ofmArr = {kRange, bRange, hRange, wRange};
					vol_t s = (*ofmArr).size();
					ofmLayout.update(*ofmArr);
					if(fmap_K || k == 0){
						if(s == 0){
							*ifmArr = emptyRange;
						}else{
							*ifmArr = *ofmArr;
							layer.ofm_to_ifm(*ifmArr);
							ifmLayout.update(*ifmArr);

						}
						++ifmArr;
					}
					if(hasWgt && (wgt_B || b == 0) && h == 0 && w == 0){
						if(s == 0){
							*wgtArr = emptyRange;
						}else{
							*wgtArr = *ofmArr;
							layer.ofm_to_wgt(*wgtArr);
							if(!wgt_B) wgtArr->b = {0, 1};
							wgtLayout.update(*wgtArr);
						}
						++wgtArr;
					}
					++ofmArr;
				}
			}
		}
	}
	assert(ifmArr == ifmLayout.rangeArr.get() + ifmLayout.rangeLength());
	assert(wgtArr == wgtLayout.rangeArr.get() + wgtLayout.rangeLength());
	assert(ofmArr == ofmLayout.rangeArr.get() + ofmLayout.rangeLength());

	// Multiply size of ifmLayout for eltwise
	auto* eltLayer = dynamic_cast<const EltwiseLayer*>(&layerT.layer());
	if(eltLayer != nullptr){
		ifmLayout.sizeMult(eltLayer->get_workload().N);
	}

	delete[] arrs[0];
	delete[] arrs[1];
	delete[] arrs[2];
	delete[] arrs[3];
}
void FastLayerEngine::initLayouts(PlaceSch& place, const Node& layerT, const fmap_shape& ofmShape, len_t B) const {
	using plen_t = PartSch::partlen_t;

	const PartSch& part = place.part;
	const Layer& layer = layerT.layer();
	bool fmap_K = layer.fmap_channel_rel();
	bool hasWgt = layer.weight_size() > 0;
	bool wgt_B = layerT.hasWgtPrevs();

	len_t* arrs[4];
	arrs[0] = part_intv(ofmShape.c, part.K);
	arrs[1] = part_intv(B, part.B);
	arrs[2] = part_intv(ofmShape.h, part.H);
	arrs[3] = part_intv(ofmShape.w, part.W);
	fmap_range::dim_range kRange, bRange, hRange, wRange;
	fmap_range emptyRange({ 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 });

	auto& ofmLayout = static_cast<StdULayout&>(place.getOfmL());
	auto& ifmLayout = static_cast<StdDataLayout&>(place.getIfmL());
	auto& wgtLayout = static_cast<StdDataLayout&>(place.getWgtL());
	ofmLayout.setDims(part.K, part.B, part.H, part.W);
	if (fmap_K)
		ifmLayout.setBcast(1, 1);
	else
		ifmLayout.setBcast(part.K, part.B * part.H * part.W);

	if (hasWgt) {
		if (wgt_B)
			wgtLayout.setBcast(part.H * part.W, 1);
		else
			wgtLayout.setBcast(part.B * part.H * part.W, 1);
	}
	auto* ofmArr = ofmLayout.rangeArr.get();
	auto* ifmArr = ifmLayout.rangeArr.get();
	auto* wgtArr = wgtLayout.rangeArr.get();
	for (plen_t k = 0; k < part.K; ++k) {
		kRange = { arrs[0][k], arrs[0][k + 1] };
		for (plen_t b = 0; b < part.B; ++b) {
			bRange = { arrs[1][b], arrs[1][b + 1] };
			for (plen_t h = 0; h < part.H; ++h) {
				hRange = { arrs[2][h], arrs[2][h + 1] };
				for (plen_t w = 0; w < part.W; ++w) {
					wRange = { arrs[3][w], arrs[3][w + 1] };
					*ofmArr = { kRange, bRange, hRange, wRange };
					vol_t s = (*ofmArr).size();
					ofmLayout.update(*ofmArr);
					if (fmap_K || k == 0) {
						if (s == 0) {
							*ifmArr = emptyRange;
						}
						else {
							*ifmArr = *ofmArr;
							layer.ofm_to_ifm(*ifmArr);
							ifmLayout.update(*ifmArr);

						}
						++ifmArr;
					}
					if (hasWgt && (wgt_B || b == 0) && h == 0 && w == 0) {
						if (s == 0) {
							*wgtArr = emptyRange;
						}
						else {
							*wgtArr = *ofmArr;
							layer.ofm_to_wgt(*wgtArr);
							if (!wgt_B) wgtArr->b = { 0, 1 };
							wgtLayout.update(*wgtArr);
						}
						++wgtArr;
					}
					++ofmArr;
				}
			}
		}
	}
	assert(ifmArr == ifmLayout.rangeArr.get() + ifmLayout.rangeLength());
	assert(wgtArr == wgtLayout.rangeArr.get() + wgtLayout.rangeLength());
	assert(ofmArr == ofmLayout.rangeArr.get() + ofmLayout.rangeLength());

	// Multiply size of ifmLayout for eltwise
	auto* eltLayer = dynamic_cast<const EltwiseLayer*>(&layerT.layer());
	if (eltLayer != nullptr) {
		ifmLayout.sizeMult(eltLayer->get_workload().N);
	}

	delete[] arrs[0];
	delete[] arrs[1];
	delete[] arrs[2];
	delete[] arrs[3];
}
void StdLayerEngine::calcNoC(NoC& noc, const PlaceSch& place, LNode* curNode, bool unordered) const{
	noc.reset();
	const Node& layerT = curNode->layert;
	len_t B = curNode->num_batch;
	bool wgt_B = layerT.hasWgtPrevs();
	len_t curC;

	// Fetch weight first.
	if(wgt_B){
		// Fetch each prev layer from its ofmap/mem layout
		curC = 0;
		const auto& prevs = layerT.getWgtPrevs();
		FOR_BITSET(it, prevs){
			lid_t prev = it;
			len_t prevC = network->getNode(prev).layer().ofmap_shape().c;
			if(curNode->get_dirp_set().contains(prev)){
				LNode* fromNode = (*(curNode->lnodeList))[prev];
				const auto& fromLayout = fromNode->get_place_sch().getOfmL();
				noc.betweenLayout(fromLayout, place.getWgtL(), curC, fromNode->num_batch, B, unordered);
			}else{
				// TODO: Change to last layer's memLayout.
				noc.fromRemoteMem(place.getWgtL(), curC, curC + prevC, NoC::interleave?-5:(*(curNode->lnodeList))[prev]->get_nearest_dram());
			}
			curC += prevC;
		}
		assert(curC == layerT.layer().weight_shape().c);
	}else{
		noc.fromRemoteMem(place.getWgtL(), NoC::interleave ? -5 : curNode->get_nearest_dram());
		noc /= (curNode->wgt_bgrp / B);
	}
	if (place.fetch) {
		// Currently, no ifm multiple fetch, so just mult weight fetch.
		assert(place.fetch.ifmFetch == 1);
		if (place.fetch.wgtFetch > 1) {
			noc *= place.fetch.wgtFetch;
		}
	}

	// Identify eltwise first
	len_t elt_K = 0, cur_N = 0;
	if(REF_IS_INSTANCE(layerT.layer(), EltwiseLayer)){
		elt_K = layerT.layer().ofmap_shape().c;
	}

	// Fetch external data from remote MEM
	curC = layerT.get_external_C();
	noc.fromRemoteMem(place.getIfmL(), 0, curC, NoC::interleave ? -5 : curNode->get_nearest_dram());
	if(elt_K > 0){
		if(curC == elt_K){
			curC = 0;
			++cur_N;
		}else{
			// TODO: here we have asserted input_C < elt_K for eltwise layer, so its fine.
			assert(curC <= elt_K);
		}
	}

	// Fetch each prev layer from its ofmap/mem layout
	const auto& prevs = layerT.getIfmPrevs();
	FOR_BITSET(it, prevs){
		lid_t prev = it;
		len_t prevC = network->getNode(prev).layer().ofmap_shape().c;
		if(curNode->get_dirp_set().contains(prev)){
			LNode* fromNode = (*(curNode->lnodeList))[prev];
			const auto& fromLayout = fromNode->get_place_sch().getOfmL();
			noc.betweenLayout(fromLayout, place.getIfmL(), curC, fromNode->num_batch, B, unordered);
		}else{
			// TODO: Change to last layer's memLayout.
			noc.fromRemoteMem(place.getIfmL(), curC, curC + prevC, NoC::interleave ? -5 : (*(curNode->lnodeList))[prev]->get_nearest_dram());
		}
		curC += prevC;
		if(elt_K > 0){
			if(curC == elt_K){
				curC = 0;
				++cur_N;
			}else{
				// TODO: here we have asserted input_C < elt_K for eltwise layer, so its fine.
				assert(curC <= elt_K);
			}
		}
	}
	if(elt_K > 0) curC = elt_K * cur_N;
	assert(curC == layerT.layer().real_ifmap_shape().c);

	// Save to remote mem if necessary
	if(curNode->to_dram){
		noc.toRemoteMem(place.getOfmL(), NoC::interleave ? -5 : curNode->get_nearest_dram());
	}
}

void StdLayerEngine::calcNoC(NoC& noc, const PlaceSch& place, const Light_placement& light_place, LNode* curNode, bool unordered) const {
	noc.reset();
	const Node& layerT = curNode->layert;
	len_t B = curNode->num_batch;
	bool wgt_B = layerT.hasWgtPrevs();
	len_t curC;
	// Fetch weight first.
	if (wgt_B) {
		// Fetch each prev layer from its ofmap/mem layout
		curC = 0;
		const auto& prevs = layerT.getWgtPrevs();
		FOR_BITSET(it, prevs) {
			lid_t prev = it;
			len_t prevC = network->getNode(prev).layer().ofmap_shape().c;
			if (curNode->get_dirp_set().contains(prev)) {
				LNode* fromNode = (*(curNode->lnodeList))[prev];
				const auto& fromLayout = fromNode->get_place_sch().getOfmL();
				noc.betweenLayout(fromLayout, place.getWgtL(), curC, fromNode->num_batch, B, unordered);
			}
			else {
				// TODO: Change to last layer's memLayout.
				
				noc.fromRemoteMem(place.getWgtL(), curC, curC + prevC, light_place.layer_DRAM.at(prev)[2]);
			}
			curC += prevC;
		}
		assert(curC == layerT.layer().weight_shape().c);
	}
	else {
		noc.fromRemoteMem(place.getWgtL(), light_place.layer_DRAM.at(curNode->getLayer().getid())[1]);
		noc /= (curNode->wgt_bgrp / B);
	}
	if (place.fetch) {
		// Currently, no ifm multiple fetch, so just mult weight fetch.
		assert(place.fetch.ifmFetch == 1);
		if (place.fetch.wgtFetch > 1) {
			noc *= place.fetch.wgtFetch;
		}
	}
	// Identify eltwise first
	len_t elt_K = 0, cur_N = 0;
	if (REF_IS_INSTANCE(layerT.layer(), EltwiseLayer)) {
		elt_K = layerT.layer().ofmap_shape().c;
	}

	// Fetch external data from remote MEM
	curC = layerT.get_external_C();
	noc.fromRemoteMem(place.getIfmL(), 0, curC, light_place.layer_DRAM.at(curNode->getLayer().getid())[0]);
	if (elt_K > 0) {
		if (curC == elt_K) {
			curC = 0;
			++cur_N;
		}
		else {
			// TODO: here we have asserted input_C < elt_K for eltwise layer, so its fine.
			assert(curC <= elt_K);
		}
	}

	// Fetch each prev layer from its ofmap/mem layout
	const auto& prevs = layerT.getIfmPrevs();
	FOR_BITSET(it, prevs) {
		lid_t prev = it;
		len_t prevC = network->getNode(prev).layer().ofmap_shape().c;
		if (curNode->get_dirp_set().contains(prev)) {
			LNode* fromNode = (*(curNode->lnodeList))[prev];
			const auto& fromLayout = fromNode->get_place_sch().getOfmL();
			noc.betweenLayout(fromLayout, place.getIfmL(), curC, fromNode->num_batch, B, unordered);
		}
		else {
			// TODO: Change to last layer's memLayout.
			noc.fromRemoteMem(place.getIfmL(), curC, curC + prevC, light_place.layer_DRAM.at(prev)[2]);
		}
		curC += prevC;
		if (elt_K > 0) {
			if (curC == elt_K) {
				curC = 0;
				++cur_N;
			}
			else {
				// TODO: here we have asserted input_C < elt_K for eltwise layer, so its fine.
				assert(curC <= elt_K);
			}
		}
	}
	if (elt_K > 0) curC = elt_K * cur_N;
	assert(curC == layerT.layer().real_ifmap_shape().c);

	// Save to remote mem if necessary
	if (curNode->to_dram) {
		noc.toRemoteMem(place.getOfmL(), light_place.layer_DRAM.at(curNode->getLayer().getid())[2]);
	}
}

bool LayerScheme::isValid() const{
	return totCost.isValid();
}
