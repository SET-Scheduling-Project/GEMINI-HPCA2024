#ifndef NNS_H
#define NNS_H

#include "network.h"
#include "util.h"

extern const Network darknet19;

extern const Network googlenet;

extern const Network inception_resnet_v1;

extern const Network resnet50;
extern const Network resnet101;
extern const Network resnet152;

//extern Network vgg16;
extern const Network vgg19;

extern const Network zfnet;
extern const Network alexnet;
extern const Network densenet;

extern const Network gnmt;
extern const Network lstm;

extern const Network transformer;
extern const Network transformer_cell;

extern const Network PNASNet;
extern const Network resnext50;

Network gen_network(std::string model_name);
#endif // NNS_H
