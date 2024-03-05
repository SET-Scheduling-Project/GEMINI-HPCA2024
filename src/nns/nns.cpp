
// #include <string.h>
#include <iostream>
#include <fstream>
using namespace std;
#include "network.h"
#include "layer.h"
#include "network.h"
#include "nns/nns.h"
#include "../json/json.hpp"
using namespace nlohmann;
using ordered_json = nlohmann::ordered_json;

typedef TransposeLayer::dim _dim;


void err_layer_not_supported(const string& operation){
	std::cerr << "The " << operation << " operation is not implemented in layer.cpp or layer.h!" << std::endl;
	throw std::logic_error("Layer type not supported.");
}

/* TODO: 
	1. Support more layer-kind.
	2. Loose inspection of input dimension in function `add` in `network.cpp`.
 */
Network gen_network(string model_name){
	Network n;
	string ir_path="/home/pengsen/projects/stschedule/src/front_end_IRs/" + model_name + ".json";
	std::ifstream f(ir_path);
	ordered_json total_ir = ordered_json::parse(f);
	map<string, Network::lid_t> layer_map;
    
	string reshape_pre;
	vector<int> reshape_input_shape;
	bool is_reshape = false;
	for (auto layer_ir:total_ir.items()){
		ordered_json layer = layer_ir.value();
		vector<string> previous_layer = layer["previous_layer"];
		vector<string> next_layer = layer["next_layer"];
		string operation = layer["operation"];
		vector<int> output_shape = layer["output_shape"];
		string name = layer["name"];
		cout<<name<<"  "<<operation<<endl;
		Network::lid_t cur_id;
		vector<vector<int>> input_shape_list = layer["input_shape"];
        string input_dtype = layer["input_dtype"][0];
		string device =  layer["device"];
		bwidth_t bitwidth = (input_dtype == "float16" || input_dtype == "int16")?16:8;
		cout<<name<<"  "<<operation<<endl;
		// TODO: how to handle two or more inputs.
		vector<int> input_shape = input_shape_list[0];

		if(operation != "dense"){
			if(input_shape.size()!=4){
					for(int i =0;i<4-input_shape.size();i++){
						input_shape.insert(input_shape.begin(),1);
					}
				}
			if(output_shape.size()!=4){
				for(int i =0;i<4-output_shape.size();i++){
					output_shape.insert(output_shape.begin(),1);
				}
			} 
		}

		/*
		previous:
		1、某一层的输出
		   1.tensor/vector: 添加进prevs
		   2.scalar: 不添加进prevs，scalar来自寄存器，不占用l2 buffer
		2、constant
		   1.tensor/vector: 添加进ext_data,vector的H和W=1
		   2.scalar: 不添加进prevs，scalar来自指令配置，不占用l2 buffer
		*/

		vector<Network::lid_t> prevs;
		vector<InputData> ext_data; 
		int pre_num = 0;
		for(int i=0; i<previous_layer.size();i++){
			string cur_prev = previous_layer[i]; 
			vector<int> scalar_shape={1,1,1,1};

			if(input_shape_list[i] == scalar_shape || previous_layer[i].find("weight")!=string::npos ){   // 如果prevs_layer为scalar,则不进行处理 
				continue;
			}
			else if(previous_layer[i].find("constant")==string::npos && previous_layer[i].find("input")==string::npos && previous_layer[i].find("cls")==string::npos){   // TODO 增加到对vector的处理 另外，ext_data的batch不一致（N,C）/（1,C）
				prevs.push_back(layer_map[cur_prev]);
			}
			else{   // constant(包含input) tensor/vector,作为ext_data
				InputData data(previous_layer[i], fmap_shape(input_shape_list[i][1], input_shape_list[i][2], input_shape_list[i][3]));	 // TODO 要求IR提供[N,C,H,W]和input_shape_list
				ext_data.push_back(data);    // TODO	
			}
		}
		bool is_Elt = false;   // TODO 涉及到mul,sum_sub为true
		if(operation == "conv2d" || operation == "conv2d_group"){			
            vector<int> kernel_size = layer["kernel_size"];
			vector<int> padding = layer["padding"];
			vector<int> stride = layer["stride"];
	        if(previous_layer[0].find("input")!=string::npos){
				InputData input("input0", fmap_shape(input_shape[1], input_shape[2], input_shape[3]));	
				cur_id = n.add(NLAYER(name, Conv, C=input_shape[1], K=output_shape[1], H=output_shape[2], W=output_shape[3],R=kernel_size[0],sH=stride[0]),{},bitwidth,{input});
			} else {
				cur_id = n.add(NLAYER(name, Conv, C=input_shape[1], K=output_shape[1], H=output_shape[2], W=output_shape[3],R=kernel_size[0],S=kernel_size[1], sH=stride[0]),prevs);
			}
		} else if(operation == "conv2d_dilation"){
            vector<int> kernel_size = layer["kernel_size"];
			vector<int> padding = layer["padding"];
			vector<int> stride = layer["stride"];
			vector<int> dilation =  layer["dilation"];
			if(previous_layer[0].find("input")!=string::npos){
				InputData input("input0", fmap_shape(input_shape[1], input_shape[2], input_shape[3]));	
				cur_id = n.add(NLAYER(name, Conv, C=input_shape[1], K=output_shape[1], H=output_shape[2], W=output_shape[3],R=kernel_size[0],sH=stride[0]),{},bitwidth,{input});
			} else{
				cur_id = n.add(NLAYER(name, Conv, C=input_shape[1], K=output_shape[1], H=output_shape[2], W=output_shape[3],R=kernel_size[0],sH=stride[0]),prevs);
			}
		} else if(operation == "dense"){			
			if(is_reshape == true){
				prevs = {layer_map[reshape_pre]};
				input_shape = reshape_input_shape;
			}
			if(input_shape.size() == 2){
                 input_shape.push_back(1);
				 input_shape.push_back(1);
			}
			if(output_shape.size() == 2){
                 output_shape.push_back(1);
				 output_shape.push_back(1);
			}
			const Layer* l = NLAYER(name, FC, C=input_shape[1], K=output_shape[1], IH=input_shape[2], IW=input_shape[3]);
			cur_id = n.add(l,prevs);
		} else if(operation == "reshape"  && total_ir[next_layer[0]]["operation"] == "dense"){
			reshape_pre = previous_layer[0];
			reshape_input_shape = input_shape;
			is_reshape = true;
		} else if(operation == "permute"){
			// TODO: modify the initial params. dims -> order but not axes
			const Layer* l = NLAYER(name, Transpose, K=output_shape[1], H=output_shape[2],W=output_shape[3], order[_dim::C]=layer["axes"][_dim::C], order[_dim::H]=layer["axes"][_dim::H], order[_dim::W]=layer["axes"][_dim::W]);
			cur_id = n.add(l,prevs);
		} else if (operation == "pooling") {
			// TODO: unknown about how many kinds of operation we have.
		}
	}
    auto max_buf_size=0;
	for(int i=0;i<n.len();i++){
		cout<<n[i].layer().get_name()<<"  "<< n[i].layer().real_ifmap_shape()<<"   "<< n[i].layer().ofmap_shape()<<"  "<<n[i].layer().weight_size()<<endl;
		auto buf_size = n[i].layer().real_ifmap_shape().tot_size(1)+n[i].layer().weight_size()+10*1024;
		if(buf_size>max_buf_size){
			max_buf_size = buf_size;
		}
	
	}
	cout<<"max_buf_size:"<<max_buf_size<<endl;
	return n;
	
}