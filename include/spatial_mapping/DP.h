#include "spatial_mapping/segmentation.h"
#include "spatial_mapping/light_placement.h"

class DP{
    std::vector<Segment_scheme> prefix_cost;
    lid_t layer_num;
    void transfer(lid_t prev, lid_t now);
public:
    DP();
};
