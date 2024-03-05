#include "spatial_mapping/light_placement.h"
#include "cluster.h"
#include "schnode.h"
#include "layerengine.h"
#include <cassert>
#include <random>
#include "util.h"

FastLayerEngine* Light_placement::fast_buffer = nullptr;

Light_partition::Light_partition(){

}

void Light_partition::init(const fmap_shape& shape, len_t _batch, vol_t _layerno){
    PartSch partSch;
    batch = _batch;
    auto partIter = partEngine.init(siz, batch, network->getNode(_layerno), partSch, 0);
    allpartition.clear();
    do {
        if(partSch.B<=batch&&partSch.K<=shape.c&&partSch.H<=shape.h&&partSch.W<=shape.w){
            std::vector<len_t> fac={partSch.B, partSch.K, partSch.H, partSch.W};
            allpartition.push_back(fac);
        }
	} while (partIter.nextPart());
    assert(!allpartition.empty());
}

void Light_partition::re_init(const fmap_shape& shape, vol_t size, PartSch part) {
    PartSch partSch;
    siz = size;
    b = part.B;
    c = part.K;
    h = part.H;
    w = part.W;
    auto partIter = partEngine.init(siz, batch, network->getNode(layerno), partSch, 0);
    allpartition.clear();
    do {
        if (partSch.B <= batch && partSch.K <= shape.c && partSch.H <= shape.h && partSch.W <= shape.w) {
            std::vector<len_t> fac = { partSch.B, partSch.K, partSch.H, partSch.W };
            allpartition.push_back(fac);
        }
    } while (partIter.nextPart());
    assert(!allpartition.empty());
}

Light_partition::Light_partition(const fmap_shape& shape, cidx_t _siz, len_t _batch, vol_t _layerno){
    PartSch partSch;
    siz = _siz;
    layerno = _layerno;
    batch = _batch;
    auto partIter = partEngine.init(siz, batch, network->getNode(_layerno), partSch, 0);
    do {
        if(partSch.B<=batch&&partSch.K<=shape.c&&partSch.H<=shape.h&&partSch.W<=shape.w){
            std::vector<len_t> fac={partSch.B, partSch.K, partSch.H, partSch.W};
            allpartition.push_back(fac);
        }
	} while (partIter.nextPart());

    if(!allpartition.empty())change();
}

Light_partition::Light_partition(vol_t _b, vol_t _c, vol_t _h, vol_t _w, vol_t _siz, vol_t _layerno):
    b(_b),c(_c),h(_h),w(_w),siz(_siz),layerno(_layerno){

}

Light_placement::Layer_partition::Layer_partition(){

}

Light_placement::Layer_partition::Layer_partition(lid_t _layerno, cidx_t _partno):
    layerno(_layerno),partno(_partno){}

Light_placement::Light_placement(){}

/*Light_placement::Light_placement(SchNode* segment){
    assert(segment->get_type()==SchNode::NodeType::S);
    auto pt = dynamic_cast<SCut*>(segment);
}*/

Light_placement::Light_placement(LTreeNode* _node){
    node = _node;
    const auto& cnodes = node->get_children();
    layer_num = 0;
	cidx_t cnum = static_cast<cidx_t>(cnodes.size());
	assert(cnum > 0);
	utime_t* tlist = new utime_t[cnum];
	utime_t* cur_item = tlist;
	for(auto child: cnodes){
		*(cur_item++) = child->get_utime();
	}
    Cluster cluster(0,Cluster::xlen*Cluster::ylen);
	auto allocRes = cluster.try_alloc(tlist, cnum);
	delete[] tlist;
    
    int allocidx=0;
    layers = node->layers();
    FOR_BITSET(layerno, node->layers()){
        for(int j=0;j<allocRes[allocidx+1]-allocRes[allocidx];++j){
            placement.emplace_back(layerno,j);
            layer_scheme[layerno].insert(allocRes[allocidx]+j);
        }
        partition.emplace_back(network->getNode(layerno).layer().ofmap_shape(), allocRes[allocidx+1]-allocRes[allocidx], node->get_bgrp_size(), layerno);
        ++allocidx;
        ++layer_num;
    }
}

const std::vector<Light_partition>& Light_placement::get_partition() const {
    return partition;
}

const std::unordered_map<lid_t, std::unordered_set<cidx_t>>& Light_placement::get_layer_scheme() const {
    return layer_scheme;
}
const std::unordered_map<lid_t, FetchSch>& Light_placement::get_fetch_scheme()const {
    return fetch_scheme;
}

const std::vector<Light_placement::Layer_partition>& Light_placement::get_placement() const {
    return placement;
}

void Light_placement::swap(cidx_t x, cidx_t y){
    if (placement[x].layerno != placement[y].layerno) {
        layer_scheme[placement[x].layerno].erase(x);
        layer_scheme[placement[x].layerno].insert(y);
        layer_scheme[placement[y].layerno].erase(y);
        layer_scheme[placement[y].layerno].insert(x);
    }
    std::swap(placement[x], placement[y]);
}

void Light_partition::change(){
    static std::mt19937_64 gen(233);
    int ch=gen()%allpartition.size();
    b=allpartition[ch][0];
    c=allpartition[ch][1];
    h=allpartition[ch][2];
    w=allpartition[ch][3];
}

void Light_placement::change_core(int cnt) {
    if (cnt == 16) {
        return;//exit
    }
    static std::mt19937_64 gen(165);
    cidx_t layer1, layer2,layer1_size,layer2_size;
    cidx_t temp_layer1, temp_layer2;
    int cnt1 = 0;
    do {
        temp_layer1 = gen() % layer_num;
        temp_layer2 = gen() % layer_num;
        layer1 = partition[temp_layer1].layerno;
        layer2 = partition[temp_layer2].layerno;
        cnt1++;
        if (cnt1 == 16) {
            mutate(Cluster::xlen * Cluster::ylen);
            return;
        }
    } while (layer1 == layer2 || layer_scheme[layer1].size()==1);

    layer1_size = layer_scheme[layer1].size();//core number
    layer2_size = layer_scheme[layer2].size();
    std::unordered_set<cidx_t>::iterator iter1 = layer_scheme[layer1].begin();
    cidx_t temp1 = *iter1;
    cidx_t rand_core_id = gen() % layer1_size;
    for (cidx_t i = 0; i < rand_core_id; ++i) {
        temp1 = *(++iter1);
    }
    layer1_size--;
    layer2_size++;
    
    Cluster c1(0, layer1_size);
    Cluster c2(0, layer2_size);
    PartSch part1 = fast_buffer->fast_get_part(c1, network->getNode(layer1), partition[temp_layer1].batch);
    if (part1.size() == 0) {
        change_core(++cnt);
        return;
    }
    PartSch part2 = fast_buffer->fast_get_part(c2, network->getNode(layer2), partition[temp_layer2].batch);
    if (part2.size() == 0) {
        change_core(++cnt);
        return;
    }
    layer_scheme[layer1].erase(temp1);
    placement[temp1].layerno = layer2;
    partition[temp_layer1].re_init(network->getNode(layer1).layer().ofmap_shape(), layer1_size, part1);
    partition[temp_layer2].re_init(network->getNode(layer2).layer().ofmap_shape(), layer2_size, part2);
    for (auto core_id : layer_scheme[layer1]) {
        if (placement[core_id].partno > placement[temp1].partno) {
            placement[core_id].partno--;
        }
    }
    cidx_t partnp_temp = gen() % layer2_size;
    for (auto core_id : layer_scheme[layer2]) {
        if (placement[core_id].partno >= partnp_temp) {
            placement[core_id].partno++;
        }
    }
    layer_scheme[layer2].insert(temp1);
    placement[temp1].partno = partnp_temp;
}

void Light_placement::change_DRAM() {
    static std::mt19937_64 gen(165);
    cidx_t layer1;
    cidx_t temp_layer1;
    mlen_t new_ddr = -1;
    int data = -1;
    do {
        temp_layer1 = gen() % layer_num;
        layer1 = partition[temp_layer1].layerno;
        data = gen() % 3;
        new_ddr = gen() % (NoC::DRAM_num+1);
        if (new_ddr == NoC::DRAM_num) {
            new_ddr = -2;
        }
    } while (new_ddr == layer_DRAM[layer1][data]|| layer_DRAM[layer1][data]==-1);
    layer_DRAM[layer1][data] = new_ddr;
}


void Light_placement::mutate(cidx_t dist){
    static std::mt19937_64 gen(165);
    int type = gen() % 11;
    if(type<6){
        cidx_t x,y;
        do{
            x=gen()%placement.size();
            y=gen()%placement.size();
        }
        while(dis(Cluster::get_pos(x),Cluster::get_pos(y))>dist);
        swap(x,y); 
    }
    else if (type < 8 && layer_num>1) {
        change_core();
    }

    else if (type < 9) {
        change_DRAM();
    }
    else{
        lid_t x=gen()%partition.size();
        partition[x].change();
    }
}

void Light_placement::crossover(const Light_placement &other){
    static std::mt19937_64 gen(277);
    if(gen()&1){
        placement=other.get_placement();
    }
    for(int i=0;i<partition.size();++i){
        if(gen()&1){
            partition[i]=other.partition[i];
        }
    }
}

void Light_placement::print(){
    std::vector<std::vector<Layer_partition> > matrix(Cluster::ylen, std::vector<Layer_partition>(Cluster::xlen));
    cidx_t core_id = 0;
    for(auto &part: placement){
        auto pos = Cluster::get_pos(core_id);
        matrix[pos.y][pos.x] = part;
        ++core_id;
    }
    for(cidx_t i=Cluster::ylen-1;i>=0;--i,puts("")){
        for(cidx_t j=0;j<Cluster::xlen;++j){
            printf("(%d,%d),",matrix[i][j].layerno,matrix[i][j].partno);
        }
    }
}