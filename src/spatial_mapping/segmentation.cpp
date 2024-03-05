#include "spatial_mapping/segmentation.h"
#include "ltreenode.h"
#include "schnode.h"
#include "cluster.h"
#include "util.h"
#include "debug.h"
#include <cassert>
#include <random>
#include <time.h>

std::vector<std::pair<vol_t, cidx_t>> ave_core;

Segment_scheme::Segment_scheme(const std::vector<Segment_scheme::lid_t> &_order, const Bitset &_border, const std::vector<len_t> &_batch, len_t _batch_num){
    batch_num = _batch_num;
    layer_num = _order.size()-1;
    ltree = nullptr;
    schtree = nullptr;
    schtree_spm = nullptr;
    schtree_fast = nullptr;
    subschtree = nullptr;
    non_conv_size = 0;
    order = _order;
    border = _border;
    batch = _batch;
    id_to_order.resize(layer_num+1);
    for(lid_t i=0;i<=layer_num;++i){
        id_to_order[order[i]]=i;
    }
    for(int i=0;i<=layer_num;++i){
        isNew.set(i);
    }
}

Segment_scheme::Segment_scheme(lid_t layer_num_, len_t _batch_num, Bitset& input_not){
    batch_num=_batch_num;
    not_conv_layers = input_not;
    ltree = nullptr;
    schtree = nullptr;
    schtree_spm = nullptr;
    schtree_fast = nullptr;
    subschtree = nullptr;
    //oldSchtree = nullptr;
    non_conv_size = not_conv_layers.count();
    //for(lid_t i=0; i<layer_num_-1; ++i){
    //    border.set(i);
    //}
    order.resize(layer_num_);
    id_to_order.resize(layer_num_);
    for(lid_t i=0;i<layer_num_;++i){
        order[i]=i;
        id_to_order[i]=i;
    }
    layer_num = layer_num_ - 1;
    batch.resize(layer_num_);
    //for(lid_t i=0;i< layer_num_;++i){
    //    batch[i]=1;
    //    isNew.set(i);
    //}
    batch[0]=1;
    isNew.set(0);
}

void Segment_scheme::clear(){
    if(schtree) delete schtree;
    if(subschtree) delete subschtree;
    if(schtree_spm) delete schtree_spm;
    if(ltree) delete ltree;
}

void Segment_scheme::copy_from(const Segment_scheme& rhs){
    ltree = rhs.ltree ? rhs.ltree->copy() : nullptr;
    schtree = rhs.schtree ? rhs.schtree->copy() : nullptr;
    schtree_spm = rhs.schtree_spm ? rhs.schtree_spm->copy() : nullptr;
    subschtree = rhs.subschtree ? rhs.subschtree->copy() : nullptr;
    schtree_fast = rhs.schtree_fast ? rhs.schtree_fast->copy() : nullptr;
}

Segment_scheme::~Segment_scheme(){
    clear();
}

void Segment_scheme::flip(lid_t left, bool use_front_batch){
    if(border.contains(left)){
        // using front batch means fused segments use batch of left segment, otherwise right segment 
        if(use_front_batch){
            batch[left+1]=0;
        }
        else{
            lid_t i;
            for(i=left;i>=0;--i){
                if(batch[i]){
                    break;
                }
            }
            assert(i>=0);
            batch[i]=batch[left+1];
            batch[left+1]=0;
        }
    }
    else{
        lid_t i;
        for(i=left;i;--i){
            if(batch[i]){
                break;
            }
        }
        assert(i>=0);
        batch[left+1]=batch[i];
    }
    border.flip(left);
}

void Segment_scheme::swap(lid_t left){
    lid_t id_left_minus = left>0?order[left - 1]:0;
    lid_t id_left = order[left];
    lid_t id_left_plus = order[left + 1];
    lid_t id_left_plus_plus = left+2<=layer_num?order[left + 2]:layer_num;
    std::swap(order[left], order[left + 1]);
    len_t tmp = id_to_order[id_left];
    id_to_order[id_left]= id_to_order[id_left_plus];
    id_to_order[id_left_plus]=tmp;
    if (not_conv_layers[id_left]&&!not_conv_layers[id_left_plus]) {
        if (left > 0 && !not_conv_layers[id_left_minus]) {
            fuse_loc.reset(left - 1);
        }  
    }
    else if (!not_conv_layers[id_left] && not_conv_layers[id_left_plus]) {
        if (left <layer_num-1 && !not_conv_layers[id_left_plus_plus]) {
            fuse_loc.reset(left+1);
        }
    }
    FOR_BITSET(i, fuse_loc) {
        assert(not_conv_layers[order[i]] || not_conv_layers[order[i + 1]]);
    }

}

void Segment_scheme::move(lid_t left, int offset){
    len_t tmp=batch[left+1];
    batch[left+1]=0;
    assert(border.contains(left));
    border.flip(left);
    border.flip(left+offset);
    batch[left+offset+1]=tmp;
}

const std::vector<lid_t>& Segment_scheme::get_order(){
    return order;
}

const Bitset& Segment_scheme::get_border(){
    return border;
}
const Bitset& Segment_scheme::get_fuse_loc() {
    return fuse_loc;
}

const std::vector<len_t>& Segment_scheme::get_batch(){
    return batch;
}


void Segment_scheme::sa_change(){
	static std::mt19937 gen(20);
    static len_t cnt = 0;
    static auto rndint = [&](int l, int r){return (int)(gen()%(r-l+1)+l);};
    
    isNew.clear();
    if(order.size() == 1){
        return;
    }
    int border_cnt = border.count();
    int segment_cnt = border.count() + 1;
    int type = rndint(1,100);
    cnt++;
    /*
    std::cout << cnt << std::endl;
    if (cnt == 116) {
        std::cout << "wrong" << std::endl;
    }*/
    const len_t max_layerid = order.size()-1;

    if(border_cnt == 0 || type<=60){
        int left = rndint(0,max_layerid-1);
        //puts("flip");
        flip(left,rndint(0,1));
        isNew.set(left);
        isNew.set(left+1);
    }
    else if (type<=65) {
        int rndseg = rndint(0, segment_cnt-1);
        if (rndseg == 0) {
            int batch_tmp = batch[0] * 2;
            if (batch_tmp > batch_num) {
                sa_change();
                //std::cout << "invalid" << std::endl;
            }
            else {
                batch[0] = batch_tmp;
                isNew.set(0);
            }
                
        }
        else {
            int tmp = border.find_nth_1(rndseg - 1) + 1;
            int batch_tmp = batch[tmp] * 2;
            if (batch_tmp > batch_num) {
                sa_change();
                //std::cout << "invalid" << std::endl;
            }
            else {
                batch[tmp] = batch_tmp;
                isNew.set(tmp);
            }
        }
    }
    else if (type<=70) {
        int rndseg = rndint(0, segment_cnt - 1);
        if (rndseg == 0) {
            int batch_tmp = batch[0] / 2;
            if (batch_tmp < 1) {
                sa_change();
                //std::cout << "invalid" << std::endl;
            }
            else {
                batch[0] = batch_tmp;
                isNew.set(0);
            }
        }
        else {
            int tmp = border.find_nth_1(rndseg - 1) + 1;
            int batch_tmp = batch[tmp] / 2;
            if (batch_tmp < 1) {
                sa_change();
                //std::cout << "invalid" << std::endl;
            }
            else {
                batch[tmp] = batch_tmp;
                isNew.set(tmp);
            }

        }
    }
    else if(type<=90){
        std::vector<lid_t> border_v;
        FOR_BITSET(i, border){
            auto &layerr = network->getNode(order[i+1]);
            if(!layerr.getPrevs().contains(order[i]))
                border_v.push_back(i);
        }
        if(border_v.size() > 0){
            lid_t loc = rndint(0, border_v.size() - 1);
            int p=border_v[loc];
            //puts("swap");
            swap(p);
            isNew.set(border_v[loc]);
            isNew.set(border_v[loc] + 1);
        }
        else{
            sa_change();
            //std::cout << "invalid" << std::endl;
        }
    }
    
    else if(type <=100){
        const int MAX_STEP = 3;
        std::vector<lid_t> border_v;
        FOR_BITSET(i, border){
            border_v.push_back(i);
        }
        int p=rndint(0,border_v.size()-1);
        len_t left = std::max(border_v[p]-MAX_STEP, p?border_v[p-1]+1:0);
        len_t right = std::min(border_v[p]+MAX_STEP, p==border_v.size()-1 ? (int)max_layerid-1 : (int)border_v[p+1]-1);
        if(left<right){
            len_t offset;
            do{
                offset = rndint(left, right)-border_v[p];
            }
            while(offset == 0);
            //printf("move %d %d\n",border_v[p],offset);
            move(border_v[p], offset);
            lid_t changed_border = border.find_nth_1(p+1);
            isNew.set(changed_border);
            isNew.set(changed_border + 1);
        }
        else{
            sa_change();
            //std::cout << "invalid" << std::endl;
        }
    }
    else if (type == 6) {
        int non_conv_id = not_conv_layers.find_nth_1( rndint(0, non_conv_size - 1));
        int order_id = id_to_order[non_conv_id];
        lid_t loc = order_id - rndint(0, 1);
        if (order_id >0 &&order_id<layer_num) {
            fuse_loc.flip(loc);
            if (!border[loc]) {
                isNew.set(loc);
            }
        }
        else if (order_id == 0) {
            fuse_loc.flip(0);
            if (!border[0]) {
                isNew.set(order_id);
            }
        }
        else if (order_id == layer_num) {
            fuse_loc.flip(layer_num-1);
            if (!border[layer_num-1]) {
                isNew.set(border.find_nth_1(border.count()-1)+1);
            }
        }
        FOR_BITSET(i, fuse_loc) {
            assert(not_conv_layers[order[i]] || not_conv_layers[order[i + 1]]);
        }
    }
}

LTreeNode* Segment_scheme::build_LTree_by_segmentation() {
    if (ltree) {
        delete ltree;
        ltree = nullptr;
    }
    Bitset layer_set;
    for(lid_t i=0;i<order.size();++i){
        layer_set.set(order[i]);
    }
    ltree=new LTreeNode(layer_set, batch_num, nullptr, LTreeNode::NodeType::T);
    ltree->isNewNode = false;
    layer_set.clear();
    lid_t l=0;
    bool tmp_new = false;
    for(lid_t i=0;i<=layer_num;++i){
        layer_set.set(order[i]);
        if (isNew[i]) {
            tmp_new = true;
        }
        if(border.contains(i)){
            LTreeNode* Scut=new LTreeNode(layer_set, batch_num, ltree, LTreeNode::NodeType::S);
            Scut->isNewNode = false;
            if (tmp_new) {
                Scut->isNewNode = true;
            }
            tmp_new = false;
            for(lid_t j=l;j<=i;){
                if (fuse_loc[j]&&!border[j]) {
                    lid_t conv = not_conv_layers[order[j]]?0:1;
                    lid_t start = j;
                    Bitset fused_layer;
                    fused_layer.set(order[j]);
                    while (fuse_loc[j] && !border[j]) {
                        if (!not_conv_layers[order[j + 1]]) {
                            conv++;
                            if (conv > 1) {
                                break;
                            }
                        }
                        assert(conv <= 1);
                        j++;
                        assert(j <= i);
                        fused_layer.set(order[j]);
                    }
                    lid_t end = j;
                    LTreeNode* Tcut = new LTreeNode(fused_layer, batch[l], Scut, LTreeNode::NodeType::T);
                    Tcut->isNewNode = false;
                    for (lid_t t = start; t <= end; ++t) {
                        LTreeNode* Lnode = new LTreeNode({ order[t] }, batch[l], Tcut, LTreeNode::NodeType::L);
                        Lnode->isNewNode = false;
                    }
                    j++;
                }
                else {
                    LTreeNode* Lnode = new LTreeNode({ order[j] }, batch[l], Scut, LTreeNode::NodeType::L);
                    Lnode->isNewNode = false;
                    assert(j <= i);
                    j++;
                    
                }
            }
            l=i+1;
            layer_set.clear();
        }
    }
    LTreeNode* Scut=new LTreeNode(layer_set, batch_num, ltree, LTreeNode::NodeType::S);
    Scut->isNewNode = false;
    if (tmp_new) {
        Scut->isNewNode = true;
    }
    for(lid_t j=l;j<=layer_num;){
        if (j < layer_num) {
            if (fuse_loc[j]) {
                lid_t conv = not_conv_layers[order[j]] ? 0 : 1;
                lid_t start = j;
                Bitset fused_layer;
                fused_layer.set(order[j]);
                while (fuse_loc[j] && !border[j]) {
                    if (!not_conv_layers[order[j + 1]]) {
                        conv++;
                        if (conv > 1) {
                            break;
                        }
                    }
                    j++;
                    assert(j <= layer_num);
                    fused_layer.set(order[j]);
                }
                lid_t end = j;
                LTreeNode* Tcut = new LTreeNode(fused_layer, batch[l], Scut, LTreeNode::NodeType::T);
                Tcut->isNewNode = false;
                for (lid_t t = start; t <= end; ++t) {
                    LTreeNode* Lnode = new LTreeNode({ order[t] }, batch[l], Tcut, LTreeNode::NodeType::L);
                    Lnode->isNewNode = false;
                }
                j++;
            }
            else {
                LTreeNode* Lnode = new LTreeNode({ order[j] }, batch[l], Scut, LTreeNode::NodeType::L);
                Lnode->isNewNode = false;
                j++;
                assert(j <= layer_num);
            }
        }
        else {
            LTreeNode* Lnode = new LTreeNode({ order[j] }, batch[l], Scut, LTreeNode::NodeType::L);
            Lnode->isNewNode = false;
            j++;
        }
    }
    layer_set.clear();
    isNew.set();
    ltree->init_root();
    return ltree;
}

bool Segment_scheme::build_subSchTree(LTreeNode* root, const Light_placement& place, bool use_placement,bool base,bool calc_noc,bool fast_mode){
    if(subschtree){
        delete subschtree;
        subschtree = nullptr;
    }
    Cluster c(0, Cluster::xlen * Cluster::ylen);
    subschtree = Cut::newNode(root, c, nullptr, use_placement, place,fast_mode, base,calc_noc);
    // use_fillin : false
    if(!subschtree->is_valid()){
        delete subschtree;
        subschtree = nullptr;
        return false;
    }
    return true;
}

bool Segment_scheme::build_SchTree(bool use_placement,SchNode* iter_res,bool base, bool calc_noc, bool fast_mode) {
    if(!use_placement && schtree){
        delete schtree;
        schtree = nullptr;
    }
    if(use_placement && schtree_spm){
        delete schtree_spm;
        schtree_spm = nullptr;
    }
    if (iter_res) {
        schtree = iter_res->copy();
        if(use_placement)dynamic_cast<Cut*>(schtree)->searchInc(ltree, fast_mode,placement);
            else schtree->searchInc(ltree, fast_mode);
    }
    else {
        Cluster c(0, Cluster::xlen * Cluster::ylen);
        use_placement ? schtree_spm = Cut::newNode(ltree, c, nullptr, placement, fast_mode,base,calc_noc) : schtree = Cut::newNode(ltree, c, nullptr,false, Light_placement(), fast_mode,base, calc_noc);
    }
    if (use_placement ? !schtree_spm->is_valid() : !schtree->is_valid()) {
        return false;
    }
    return true;
}

bool Segment_scheme::build_SchTree_fast() {
    if (schtree_fast) {
        delete schtree_fast;
        schtree_fast = nullptr;
    }
        Cluster c(0, Cluster::xlen * Cluster::ylen);
        schtree_fast = Cut::newNode(ltree, c, nullptr, true);
    
    if (!schtree_fast->is_valid()) {
        delete ltree;
        delete schtree_fast;
        ltree = nullptr;
        schtree_fast = nullptr;
        return false;
    }
    //ltree->confirm();
    return true;
}

bool sa_accept(const SchNode::SchCost &best, const SchNode::SchCost &now, double T){
	static std::default_random_engine generator;
	static std::uniform_real_distribution<double> distribution(0.0,1.0);
	if(now.cost()<=best.cost()){
		return true;
	}
	double prob = exp((best.cost()-now.cost())/(0.5*T));
	return distribution(generator) <= prob;
}

void Segment_scheme::SA_search_SPM(int iter_num, LTreeNode* root, int childno,bool base){
    time_t start_time = time(nullptr);
    auto &place = placement[childno];
    auto cutp = dynamic_cast<TCut*>(schtree);
    place.layer_DRAM = layer_Dram;
    init(place, cutp->getChildren()[childno]);
    layer_Dram = place.layer_DRAM;
    int rnd=0;
    bool best_valid = false;
    do{
        best_valid = build_subSchTree(root, place,true,base);
        if(!best_valid){
#ifdef DEBUG
            DEBUG_INFO("initial solution is invalid, trying another one");
#endif
            place.mutate(Cluster::xlen*Cluster::ylen);
        }
        else{
            //subschtree->print_struct("\t");
            /*auto &sch = subschtree->get_lnode_by_id(0)->get_place_sch();
            for(auto core: sch.getOfmL()){
                std::cout << core.first << core.second.x << " " << core.second.y << std::endl;
            }*/
        }
    }while(!best_valid);
    Light_placement iter=place, best=place;
    auto iter_cost = subschtree->get_cost();
    auto best_cost = subschtree->get_cost();
  
    double T = best_cost.cost()/10;
    double dist = 2*pow(Cluster::xlen*Cluster::ylen, 0.25);
    while(rnd<iter_num){
        if (rnd==1357) {
            //std::cout << rnd << std::endl;
        }
        place = iter;
        place.mutate(std::max(1.0,dist));
        auto valid = build_subSchTree(root, place, true, base);
        if(!valid){
            continue;
        }
        auto now_cost = subschtree->get_cost();
        if(sa_accept(iter_cost, now_cost, T)){
			iter = place;
            iter_cost = now_cost;
		}
		if(iter_cost.cost() < best_cost.cost()){
            
			best_cost = iter_cost;
            best = iter;
		}
		if(rnd%10==0){
			T *= 0.99;
            dist *= 0.998;
			//rnd = 0;
		}
        if(rnd%100==0){
            //fprintf(stderr,"SPM rnd = %d, best_cost = %lg, iter_cost = %lg, now_cost = %lg\n",rnd,best_cost.cost(),iter_cost.cost(),now_cost.cost());
        }
        ++rnd;
    }
    place=best;
    build_subSchTree(root, place,true ,base);
    //std::cout <<"normal mode dram access = " << subschtree->get_noc().get_tot_DRAM_acc() << std::endl;
    time_t end_time = time(nullptr);
    
    //subschtree->print_struct("\t");
    //std::cout << subschtree->get_cost()<<std::endl;
    //std::cout << "time = " << end_time-start_time << std::endl;
}

void mode_control(bool cluster_base, bool calc_noc, bool interleave) {
    NoC::calc_noc_control = calc_noc;
    NoC::interleave = interleave;
    SchNode::cluster_base = cluster_base;
}

void Segment_scheme::SA_search_SPM(int iter_num, bool base){
    placement.clear();
    layer_Dram.clear();
    placement.resize(ltree->get_children().size());
    mode_control(false, true, false);
    int childno = 0;
    int i = 0;
    for (auto child : ltree->get_children()) {
        
        SA_search_SPM(iter_num* child->get_children().size()*Cluster::xlen, child, childno++, base);
    }
}

void SA_search(int iter_num,bool base,bool fast_mode){
    time_t start_time = time(nullptr);
    lid_t layer_num = network->len();
    Bitset not_conv_layers;
    network->cal_not_conv(not_conv_layers);
    Segment_scheme now(layer_num, SchNode::tot_batch, not_conv_layers), best(layer_num, SchNode::tot_batch, not_conv_layers), iter(layer_num, SchNode::tot_batch, not_conv_layers);
    SchNode::SchCost best_cost, iter_cost;
    vol_t fast_cost_best, fast_cost_iter;
    do{
        iter.build_LTree_by_segmentation();
        if (!iter.build_SchTree(false,nullptr,base,!base, fast_mode)) {
            fprintf(stderr,"Retry.\n");
            std::vector<lid_t> notborder;
            for(lid_t i=0;i<layer_num-1;++i){
                if(!iter.get_border().contains(i)){
                    notborder.push_back(i);
                }
            }
            if(notborder.empty()){
                puts("Storage not enough.");
                return;
            }
            static std::mt19937_64 gen(233);
            int p;
            do{
                p=notborder[gen()%notborder.size()];
                auto root=dynamic_cast<Cut*>(iter.schtree);
                bool ok=false;
                for(auto child: root->getChildren()){
                    if(child->contains(p)&&!child->is_valid()){
                        ok=true;
                        break;
                    }
                }
                if(ok)break;
            }
            while(1);
            iter.flip(p,true);
        }
        else break;
    }while(1);
	int rnd = 0;
    iter_cost = iter.schtree->get_cost();
    best_cost = iter_cost;
    double T = best_cost.cost()/10;
    vol_t not_equal = 0;
	while(rnd<iter_num){
        //ave_core.clear();
        //ave_core.reserve(layer_num);
        now.clear();
		now = iter;
        now.copy_from(iter);
        now.sa_change();
        now.build_LTree_by_segmentation();
        bool tmp_normal = now.build_SchTree(false,iter.schtree,base,!base, fast_mode);
        //now.build_LTree_by_segmentation();
        //bool tmp_fast = now.build_SchTree_fast();
        //if (tmp_normal != tmp_fast) {
        //    std::cout << "iter = " << rnd << " normal = " << tmp_normal /*<< " fast= " << tmp_fast*/ << std::endl;
            //assert(false);
        //}
        if (!tmp_normal/*now.build_SchTree()*/) {
            continue;
        }
        SchNode::SchCost now_cost = now.schtree->get_cost();
        //vol_t fast_cost_now = now.schtree_fast->get_core_mul_data();//now.schtree_fast->get_dram_access() * now.schtree_fast->get_cost().time;
        /*
        if ((iter_cost.cost() <= now_cost.cost()) != (fast_cost_iter <= fast_cost_now)) {
            //std::cout << "iter = " << rnd << "cost_not_equal" << std::endl;
            not_equal++;
        }
*/
        //if ((iter.schtree->get_noc().get_tot_hops() <= now.schtree->get_noc().get_tot_hops()) != (fast_cost_iter <= fast_cost_now)) {
            //std::cout << "iter = " << rnd << "cost_not_equal" << std::endl;
        //   not_equal++;
        //}
		if(sa_accept(iter_cost, now_cost, T)){
            iter.clear();
			iter = now;
            iter.copy_from(now);
            iter_cost = now_cost;
            //fast_cost_iter = fast_cost_now;
        }
		if(iter_cost.cost() < best_cost.cost()){
			best_cost = iter_cost;
            best.clear();
            best = iter;
            best.copy_from(iter);
		}
		if(1){
			T *= 0.999;
			//rnd = 0;
		}
         if(rnd%50==0||iter_num-rnd<=50){
            fprintf(stderr,"rnd = %d, best_cost = %lg, iter_cost = %lg, now_cost = %lg\n",rnd,best_cost.cost(),iter_cost.cost(),now_cost.cost());
        }
        ++rnd;
	}
    best.build_LTree_by_segmentation();

    if(base){
        best.build_SchTree(false, nullptr, base,!base, fast_mode);
        best.schtree->print_struct("\t");
        std::cout << best.schtree->get_cost() << "\n";
        mode_control(true, true, false);
        best.SA_search_SPM(0,base);
        bool builded;
        builded = best.build_SchTree(true, nullptr,base,true);
        assert(builded);

        best.schtree_spm->print_struct("\t");
        
        time_t end_time = time(nullptr);
        std::cout << best.schtree_spm->get_cost() << std::endl;
        std::cout << "time = " << end_time - start_time << std::endl;

        //best_res->print_struct("\t");
        
        //std::cout << "fuse_num = " << best.get_fuse_loc().count() << std::endl;
        //std::cout << "not equal = " << not_equal << std::endl;
        //std::cout <<"fast mode dram access = "<< best.schtree_fast->get_dram_access() << std::endl;
        //std::cout <<"normal mode dram access = " << best.schtree->get_noc().get_tot_DRAM_acc() << std::endl;
        //std::cout <<"fast mode time = " << best.schtree_fast->get_cost().time << std::endl;
        return;
    }
    else {
        bool builded;
        builded = best.build_SchTree(false, nullptr, false, true);
        assert(builded);
        //assert(best.build_SchTree_fast());
        best.schtree->print_struct("\t");
        //std::cout <<"fast mode dram access = "<< best.schtree_fast->get_dram_access() << std::endl;
        //std::cout << "normal mode dram access = " << best.schtree->get_noc().get_tot_DRAM_acc() << std::endl;
        //std::cout <<"fast mode time = " << best.schtree_fast->get_cost().time << std::endl;
       

        //best_res->print_struct("\t");

        std::cout << best.schtree->get_cost() << std::endl;
        //std::cout << "fuse_num = " << best.get_fuse_loc().count() << std::endl;
        //std::cout << "not equal = " << not_equal << std::endl;
        time_t end_time = time(nullptr);
        std::cout << "time = " << end_time - start_time << std::endl;

        /*auto &sch = best.schtree->get_lnode_by_id(0)->get_place_sch();
        for(auto core: sch.getOfmL()){
            std::cout << core.first << " " << core.second.x << " " << core.second.y << std::endl;
        }*/
        start_time = time(nullptr);
        best.SA_search_SPM(iter_num);
        builded = best.build_SchTree(true);
        assert(builded);
        end_time = time(nullptr);

        best.schtree_spm->print_struct("\t");
        std::cout << best.schtree_spm->get_cost() << std::endl;
        std::cout << "time = " << end_time - start_time << std::endl;
        //std::cout << "normal mode dram access = " << best.schtree_spm->get_noc().get_tot_DRAM_acc() << std::endl;
        //SchNode* best_res = best.schtree_spm;
        //end_time = time(nullptr);

        //best_res->print_struct("\t");
        //std::cout << best_res->get_cost() << std::endl;
        //std::cout << "fuse_num = " << best.get_fuse_loc().count() << std::endl;
        //std::cout << "not equal = " << not_equal << std::endl;

        
    }
}

void overall_search(int iter_num, bool full){
    time_t start_time = time(nullptr);
    lid_t layer_num = network->len();
    Bitset not_conv_layers;
    network->cal_not_conv(not_conv_layers);
    Segment_scheme now(layer_num, SchNode::tot_batch, not_conv_layers), best(layer_num, SchNode::tot_batch, not_conv_layers), iter(layer_num, SchNode::tot_batch, not_conv_layers);
    SchNode::SchCost best_cost, iter_cost;
    vol_t fast_cost_best, fast_cost_iter;
    do{
        iter.build_LTree_by_segmentation();
        if (!iter.build_SchTree()) {
            fprintf(stderr,"Retry.\n");
            std::vector<lid_t> notborder;
            for(lid_t i=0;i<layer_num-1;++i){
                if(!iter.get_border().contains(i)){
                    notborder.push_back(i);
                }
            }
            if(notborder.empty()){
                puts("Storage not enough.");
                return;
            }
            static std::mt19937_64 gen(233);
            int p;
            do{
                p=notborder[gen()%notborder.size()];
                auto root=dynamic_cast<Cut*>(iter.schtree);
                bool ok=false;
                for(auto child: root->getChildren()){
                    if(child->contains(p)&&!child->is_valid()){
                        ok=true;
                        break;
                    }
                }
                if(ok)break;
                //std::cout << 1;
            }
            while(1);
            iter.flip(p,true);
        }
        else break;
    }while(1);
	int rnd = 0;
    iter_cost = iter.schtree->get_cost();
    best_cost = iter_cost;
    double T = best_cost.cost()/20;
    vol_t not_equal = 0;
    int stage_iter_num=iter_num;
    int spm_iter = 2;
	while(rnd<iter_num){
        now.clear();
		now = iter;
        now.copy_from(iter);
        now.sa_change();
        now.build_LTree_by_segmentation();
        bool tmp_normal = now.build_SchTree(false,iter.schtree);

        if (!tmp_normal/*now.build_SchTree()*/) {
            continue;
        }
        std::cout << rnd << std::endl;
        auto since = clock();
        now.layer_Dram.clear();
        now.SA_search_SPM(full?2*iter_num:spm_iter);
        //printf("iter = %d, time = %lf\n",spm_iter,(double)(clock()-since)/CLOCKS_PER_SEC);
        bool builded = now.build_SchTree(true); 
        assert(builded);

        SchNode::SchCost now_cost = now.schtree_spm->get_cost();
        //SchNode::SchCost now_cost = now.schtree->get_cost();

        if(sa_accept(iter_cost, now_cost, T)){
            iter.clear();
			iter = now;
            iter.copy_from(now);
            iter_cost = now_cost;
            //fast_cost_iter = fast_cost_now;
        }

		if(iter_cost.cost() < best_cost.cost()){
			best_cost = iter_cost;
            best.clear();
            best = iter;
            best.copy_from(iter);
		}
		if(1){
			T *= 0.999;
			//rnd = 0;
		}
        if(rnd%50==0||iter_num-rnd<=50){
            fprintf(stderr,"rnd = %d, best_cost = %lg, iter_cost = %lg, now_cost = %lg\n",rnd,best_cost.cost(),iter_cost.cost(),now_cost.cost());
        }
        ++rnd;
        if(iter_num-rnd<<1<=stage_iter_num){
            stage_iter_num = iter_num-rnd;
            spm_iter <<= 1;
        }
	}

    best.build_LTree_by_segmentation();
    bool builded = best.build_SchTree();
    assert(builded);
    
    //assert(best.build_SchTree_fast());
    //std::cout <<"fast mode dram access = "<< best.schtree_fast->get_dram_access() << std::endl;
    std::cout << "normal mode dram access = " << best.schtree->get_noc().get_tot_DRAM_acc() << std::endl;
    //std::cout <<"fast mode time = " << best.schtree_fast->get_cost().time << std::endl;
    time_t end_time = time(nullptr);
    
    //best_res->print_struct("\t");
    std::cout << best.schtree->get_cost()<<std::endl;
    std::cout << "fuse_num = " << best.get_fuse_loc().count() << std::endl;
    std::cout << "not equal = " << not_equal << std::endl;
    std::cout << "time = " << end_time-start_time << std::endl;

    /*auto &sch = best.schtree->get_lnode_by_id(0)->get_place_sch();
    for(auto core: sch.getOfmL()){
        std::cout << core.first << " " << core.second.x << " " << core.second.y << std::endl;
    }*/

    best.SA_search_SPM(2*iter_num);
    builded = best.build_SchTree(true);
    assert(builded);
    std::cout <<"normal mode dram access = " << best.schtree_spm->get_noc().get_tot_DRAM_acc() << std::endl;


    if(best.schtree_spm){
        delete best.schtree_spm;
        best.schtree_spm = nullptr;
    }
    builded = best.build_SchTree(true,nullptr);
    assert(builded);
    //std::cout <<"normal mode dram access = " << best.schtree_spm->get_noc().get_tot_DRAM_acc() << std::endl;

    end_time = time(nullptr);
    
    best.schtree_spm->print_struct("\t");
    std::cout << best.schtree_spm->get_cost()<<std::endl;
    //std::cout << "fuse_num = " << best.get_fuse_loc().count() << std::endl;
    //std::cout << "not equal = " << not_equal << std::endl;
    std::cout << "time = " << end_time-start_time << std::endl;
    fflush(stderr);
    /*
    best.fuse_loc.set(6);
    best.fuse_loc.reset(7);
    std::cout << get_res(best)->get_cost() << std::endl;
    get_res(best)->print_struct("\t");*/
}


len_t Segment_scheme::search_best_batch(lid_t start, lid_t end, SchNode::SchCost &result_cost,bool fast_mode) {
    /* Search the best batch_num for a segment (assert there is only one segment in this scheme). */
    // The batch of segment [start, end] is stored at batch[start];
    if (subschtree) {
        delete subschtree;
        subschtree = nullptr;
    }
    if (ltree) {
        delete ltree;
        ltree = nullptr;
    }
    // Calculate the factor of batch_num.
    len_t best_batch_size = 1;
    SchNode::SchCost current_cost, best_cost;
    border.clear();
    // Attention: `layer_num` == (number of layer) - 1.
    if (end != layer_num) border.set(end);
    batch[start] = 1;
    LTreeNode *segment;
    if (start != 0) {
        border.set(start-1);
        build_LTree_by_segmentation();
        segment = ltree->get_children()[1];
    } else {
        build_LTree_by_segmentation();
        segment = ltree->get_children()[0];
    }
    bool builded, never_valid = true;
    builded = build_subSchTree(segment, {}, false, false, false, fast_mode);
    if (!builded) {
        // TODO: how to handle invalid 1 batch.
#ifdef DEBUG
        DEBUG_INFO("Build subschtree failed. (Batch 1 invalid)");
#endif
    } else {
        never_valid = false;
        best_cost = subschtree->get_cost();
    }
    
    for (len_t current_batch_size = 2; current_batch_size <= batch_num; ++current_batch_size) {
        if (batch_num % current_batch_size == 0) {
            for (LTreeNode *child : segment->get_children()) {
                child->num_batch = current_batch_size;
            }
            segment->traverse(true);
            builded = build_subSchTree(segment, {}, false, false, false, fast_mode);

            if (!builded) {
#ifdef DEBUG
                DEBUG_INFO("Build subschtree failed.");
#endif
                continue;
            } else {
                never_valid = false;
            }
            current_cost = subschtree->get_cost();
            if (!never_valid) {
                // `best_cost.time == 0` means the best_cost has never been initialised.
                if (best_cost.time == 0 || current_cost.cost() < best_cost.cost()) {
                    best_batch_size = current_batch_size;
                    best_cost = current_cost;
                } else {
                    // Assert that the cost-batch curve goes up then down.
                    // Thus there is no need to continue to search when cost rises.
                    break;
                }
            }
        }
    }
    if (never_valid) {
        // If there is no valid batch, no modification should be done to `result_cost`.
        return 0;
    }
    result_cost = best_cost;
    return best_batch_size;
}

Segment_scheme* DP_search(int iter_num, bool base,bool fast_mode) {
    time_t start_time = time(nullptr);
    lid_t layer_num = network->len();
    len_t current_batch;
    Bitset not_conv_layers, current_border;
    network->cal_not_conv(not_conv_layers);
    Segment_scheme *iter = new Segment_scheme(layer_num, SchNode::tot_batch, not_conv_layers);
    // Initialise `batch` , otherwise it would be a 0 array.
    for (len_t &b : iter->batch) b = 1;
    
    // `dp_batch[i]` is the best batch of the best segment ended with i-th layer.
    std::vector<len_t> dp_batch(layer_num);
    // `dp_border[i]` is the best border of [0, i] layers.
    std::vector<Bitset> dp_border(layer_num);
    // `dp_cost[i]` is the lowest cost of the best segment scheme ended with i-th layer.
    std::vector<SchNode::SchCost> dp_cost(layer_num);
    SchNode::SchCost cur_cost;

    for (auto i = 0; i < layer_num; ++i) {
        // Search the best segment scheme ended with i-th layer.
        dp_batch[i] = iter->search_best_batch(i, i, dp_cost[i], fast_mode);
        // If a segment with only one layer is invalid, there would be no feasible segment.
        // assert(dp_batch[i] != 0);
        if (dp_batch[i] == 0) {
            std::cout << "dp_batch[i] is zero." << std::endl;
            return nullptr;
        }
        // Copy border (`border` has been modified according to the segment scheme in `search_best_batch`).
        dp_border[i] = iter->border;
        // When `i == 0`, there is only first layer in the segment.
        if (i == 0) continue;
        dp_border[i] |= dp_border[i-1];
        dp_cost[i] += dp_cost[i-1];

        for (auto j = i-1; j >= 0; --j) {
            // `j` is the first layer_id of the last segment [j, i].
            current_batch = iter->search_best_batch(j, i, cur_cost, fast_mode);
            if (current_batch == 0) {
#ifdef DEBUG
                // No valid batch size.
                // Assert that this is a consequence of buffer overflow.
                // Thus there is no need to add more layers into the current segment.
                DEBUG_INFO("There are too many layers a segment.");
#endif
                break;
            }
            if (j != 0) {
                // All layers are contained in more than one segments.
                cur_cost += dp_cost[j-1];
            }
            if (cur_cost.cost() < dp_cost[i].cost()) {
                dp_cost[i] = cur_cost;
                dp_border[i] = iter->border;
                if (j != 0) dp_border[i] |= dp_border[j-1];
                dp_batch[i] = current_batch;
            }
        }
        if (dp_batch[i] == 0) {
            return nullptr;
        }
    }
    // Construct best scheme's ltree using the DP searching result.
    iter->border = dp_border[layer_num-1];
    len_t segment_head = 0;
    for (len_t i = 0; i < layer_num; ++i) {
        if (iter->border[i]) {
            for (len_t j = segment_head; j <= i; ++j) {
                iter->batch[j] = dp_batch[i];
            }
            segment_head = i+1;
        }
    }
    for (len_t i = segment_head; i < layer_num; ++i) iter->batch[i] = dp_batch[layer_num-1];
    iter->build_LTree_by_segmentation();

    time_t end_time = time(nullptr);
    std::cout << "DP search process time : " << end_time - start_time << std::endl;

    // Print before-SPM result.
    bool builded;
    builded = iter->build_SchTree(false, nullptr, false, true);
    assert(builded);
#if OUTPUT_DETAIL
    std::cout << "Schtree (before SPM search): " << std::endl;
    iter->schtree->print_struct("\t");
#endif
    std::cout << std::endl << "Cost (before SPM search): " << std::endl << iter->schtree->get_cost() << std::endl;
    if(base){
        mode_control(false, true, true);
        iter->SA_search_SPM(0, base);
        builded = iter->build_SchTree(true, nullptr,base,true);
        assert(builded);
    }
    else {
        //assert(iter->build_SchTree(false, nullptr, false, true));
        iter->build_SchTree(false, nullptr, false, true,false);
        std::cout << "Schtree (before SPM search): " << std::endl;
        std::cout << "Cost (before SPM search): " << iter->schtree->get_cost() << std::endl;
        std::cout << "schtree (before SPM search):" << std::endl;
        iter->schtree->print_struct("\t");
        start_time = time(nullptr);
        mode_control(false, true, false);
        iter->SA_search_SPM(2 * iter_num);
        builded = iter->build_SchTree(true);
        assert(builded);
        end_time = time(nullptr);
        std::cout << "SPM search process time : " << end_time - start_time << std::endl;
    }
    // Print after-SPM result.
#if OUTPUT_DETAIL
    for(auto i : iter->get_order()) std::cout << i << " ";
    std::cout << std::endl;
    FOR_BITSET(i,iter->get_border()) std::cout << i << " ";
    std::cout << std::endl;
    for(auto i:iter->get_batch()) std::cout << i << " ";
    std::cout << std::endl;
#endif
    std::cout << "Schtree_spm (after SPM search):" << std::endl;
    iter->schtree_spm->print_struct("\t");
    std::cout << std::endl << "Cost (after SPM search): " << std::endl << iter->schtree_spm->get_cost() << std::endl;
    std::cout << std::endl << std::endl;
    iter->schtree_spm->write_energy_record();
    return iter;
}
