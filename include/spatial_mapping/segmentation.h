#include "network.h"
#include "bitset.h"
#include "ltreenode.h"
#include "schnode.h"
#include <vector>
//This file corresponds to the dynamic programming search for segmenting a DAG graph. It determines how to segment the graph and then perform spatial mapping optimization segment by segment.
class Segment_scheme{
    typedef Network::lid_t lid_t;
    public:
    len_t batch_num;
    len_t non_conv_size;
    len_t layer_num;
    LTreeNode* ltree;
    //SchNode* oldSchtree;
    SchNode* schtree;
    SchNode* schtree_spm;
    SchNode* subschtree;
    SchNode* schtree_fast;
    std::vector<lid_t> order;//order:layer_id
    std::vector<len_t> id_to_order;
    std::vector<len_t> batch;
    std::vector<Light_placement> placement;
    std::unordered_map<lid_t, std::vector<mlen_t>> layer_Dram;
    Bitset isNew;
    Bitset border;//layer_num-1
    Bitset fuse_loc;//layer_num-1
    Bitset not_conv_layers;
    Segment_scheme(const std::vector<lid_t> &_order, const Bitset &_border, const std::vector<len_t> &_batch, len_t _batch_num);
    Segment_scheme(lid_t layer_num, len_t _batch_num, Bitset& not_conv_layers);
    //const Segment_scheme& operator = (const Segment_scheme& rhs);
    void copy_from(const Segment_scheme& rhs);
    void clear();
    ~Segment_scheme();
    void flip(lid_t left, bool use_front_batch);
    void move(lid_t left, int offset);
    void swap(lid_t left);
    const std::vector<lid_t>& get_order();
    const std::vector<len_t>& get_batch();
    const Bitset& get_border();
    const Bitset& get_fuse_loc();
    LTreeNode* build_LTree_by_segmentation();
    bool build_SchTree(bool use_placement = false, SchNode* iter_res=nullptr, bool base=false, bool calc_noc = true,bool fast_mode=false);
    bool build_subSchTree(LTreeNode* root, const Light_placement& place, bool use_placement = true, bool base = false, bool calc_noc = true,bool fast_mode=false);
    bool build_SchTree_fast();
    void sa_change();
    void SA_search_SPM(int iter_num, bool base=false);
    void SA_search_SPM(int iter_num, LTreeNode* root, int childno, bool base = false);
    len_t search_best_batch(lid_t start, lid_t end, SchNode::SchCost &best_cost,bool fast_mode=false);
};
void SA_search(int iter_num,bool base=false, bool fast_mode = false);
void overall_search(int iter_num, bool full = false);
Segment_scheme* DP_search(int iter_num, bool base = false,bool fast_mode=false);
void mode_control(bool cluster_base, bool calc_noc, bool interleave);
extern std::vector<std::pair<vol_t, cidx_t>> ave_core;
