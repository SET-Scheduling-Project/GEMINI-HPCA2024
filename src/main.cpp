#include <iostream>
#include <fstream>
#include "schnode.h"
#include "cluster.h"
#include "util.h"
#include "noc.h"
#include "layerengine.h"
#include "spatial_mapping/segmentation.h"
#include "debug.h"

#include "nns/nns.h"

#include "ltreenode.h"

#include <cassert>
#include <time.h>

#include <string.h>
#include <math.h>
#include <random>

#include <type_traits>
#include <stdexcept>
#include <map>
#include <unordered_set>
#include <fstream>
#include <sstream>

#include <thread>
#include <mutex>

#include "json/json.h"
#include "schnode.h"

#define KB *1024

using namespace std;

struct WholeSch{
	LTreeNode* tree;
	SchNode* sch;
	WholeSch(): tree(nullptr), sch(nullptr){}
	WholeSch(LTreeNode* _tree, SchNode* _sch): tree(_tree), sch(_sch){}
	operator bool() const{
		return tree;
	}
	WholeSch copy() const{
		if(!tree) return {nullptr, nullptr};
		LTreeNode* new_tree = tree->copy();
		SchNode* new_sch = sch->copy();
		return WholeSch(new_tree, new_sch);
	}
	void del(){
		if(tree){
			delete tree;
			delete sch;
		}
	}
	void min(WholeSch& w_sch){
		if(!tree){
			tree = w_sch.tree;
			sch = w_sch.sch;
		}else if(sch->get_cost().cost() > w_sch.sch->get_cost().cost()){
			del();
			tree = w_sch.tree;
			sch = w_sch.sch;
		}else{
			w_sch.del();
		}
		w_sch.tree = nullptr;
		w_sch.sch = nullptr;
	}
	/*
	~WholeSch(){
		del();
	}
	*/
};

size_t find(const LTreeNode::node_vec& vec, LTreeNode* node){
	for(size_t i=0; i<vec.size(); ++i){
		if(vec[i] == node) return i;
	}
	assert(false);
	return vec.size();
}

void halv_bat(LTreeNode* node){
	if(!node->children.empty() && node->children.front()->num_batch == node->num_batch){
		for(auto x:node->children){
			halv_bat(x);
		}
	}
	node->num_batch /= 2;
}

void flat_bat(LTreeNode* node, len_t n_batch = 1){
	if(node->num_batch <= n_batch) return;
	for(auto x:node->children){
		flat_bat(x, n_batch);
	}
	node->num_batch = n_batch;
}

class SAEngine{
public:
	static int nrounds;
private:
	static constexpr int NUM_OP = 7;
	int cur_round;
	uint64_t num_tries;
	uint64_t cur_tries;
	std::mt19937 generator;
	ostringstream strStream;
	ostream& out;
	// Generate int between [0, to)
	int randInt(int to){
		return std::uniform_int_distribution<int>(0, to-1)(generator);
	}
	bool withProb(double prob){
		return std::uniform_real_distribution<double>(0.0, 1.0)(generator) < prob;
	}
public:
	SAEngine(std::uint32_t seed, bool directCout = false):generator(seed), out(directCout?cout:strStream){
		strStream.precision(4);
	}

	void flushBuf(){
		cout << strStream.str() << flush;
		strStream.clear();
		strStream.str("");
	}

	// sa_type: 0 -> arbitrary. 1 -> only s under top t. 2 -> only t under top t.
	LTreeNode* sa_change(LTreeNode* root, bool* valid_op, lid_t max_depth=0, int sa_type=0, int* op_type=nullptr){
		root = root->copy();
		lid_t lnum = root->layers().count();
		if(max_depth == 0) max_depth = lnum;
		if(root->height > max_depth){
			throw std::invalid_argument("The root of SA is deeper than max_depth!");
		}

		int prob[NUM_OP]={10,10,20,20,20,20,40};
		for(int i=1;i<NUM_OP;++i){
			prob[i] += prob[i-1];
		}
		//bool x[NUM_OP+1];
		//memset(x,0,NUM_OP);
		int t=NUM_OP;
		uint64_t print_tries = 30;
		cur_tries = 0;
		bool ok;
		// For convenience, we research the whole seg now.
		assert(root->get_type() == LTreeNode::NodeType::T);
		do{
			//x[t] = true;
			lid_t l = randInt(lnum);
			LTreeNode* lnode=root;
			lid_t depth = 0;
			while(lnode->t != LTreeNode::NodeType::L){
				for(auto child: lnode->children){
					if(child->layers().contains(l)){
						lnode = child;
						++depth;
						break;
					}
				}
			}
			assert(depth > 0);
			ok=false;
			do{
				int pr = randInt(prob[NUM_OP-1]);
				t=0;
				while(pr >= prob[t]) ++t;
			}while(!valid_op[t]);
			//if(x[t]) continue;
			switch (t) {
			case 0:{
				// Change with front.
				LTreeNode* front = lnode->parent;
				LTreeNode* c = lnode;
				while (front && front->children.front() == c) {
					c = front;
					front = front->parent;
				}
				if(!front) break;
				LTreeNode* lcl = c;
				size_t k = find(front->children, c);
				//cout << k << " is " << endl;
				c = front->children[k-1];
				LTreeNode* lcc = c;
				while (!c->children.empty()) {
					c = c->children.back();
				}
				lid_t x = c->layer_set.first();
				if(network->getNode(l).getPrevs().contains(x)) break;
				// Found valid front, change!
				front->stage.clear();
				// Reset path to lcl/lcc.
				while(lcl!=lnode){
					lcl->layer_set.reset(l);
					lcl->layer_set.set(x);
					lcl->stage.clear();
					for (auto z:lcl->children) {
						if(z->layer_set.contains(l)){
							lcl = z;
							break;
						}
					}
				}
				while(lcc!=c){
					lcc->layer_set.reset(x);
					lcc->layer_set.set(l);
					lcc->stage.clear();
					lcc = lcc->children.back();
				}
				size_t i = find(c->parent->children, c);
				size_t j = find(lnode->parent->children, lnode);
				c->parent->children[i] = lnode;
				lnode->parent->children[j] = c;
				swap(lnode->parent,c->parent);
				swap(lnode->num_batch,c->num_batch);
				//std::cout << (int)x << ' ' << (int)l <<std::endl;
				// Now we'll reset the whole seg.
				if(front == root){
					front->children[k]->isNewNode = true;
					front->children[k-1]->isNewNode = true;
				}else{
					while(front->parent->parent) front = front->parent;
					front->isNewNode = true;
				}
				ok=true;
			}break;
			case 1:{
				// Change with back.
				LTreeNode* back = lnode->parent;
				LTreeNode* c = lnode;
				while (back && back->children.back() == c) {
					c = back;
					back = back->parent;
				}
				if(!back) break;
				LTreeNode* lcl = c;
				size_t k = find(back->children, c);
				//cout << k << " is " << endl;
				c = back->children[k+1];
				LTreeNode* lcc = c;
				while (!c->children.empty()) {
					c = c->children.front();
				}
				lid_t x = c->layer_set.first();
				if(network->getNode(x).getPrevs().contains(l)) break;
				// Found valid back, change!
				back->stage.clear();
				// Reset path to lcl/lcc.
				while(lcl!=lnode){
					lcl->layer_set.reset(l);
					lcl->layer_set.set(x);
					lcl->stage.clear();
					for (auto z:lcl->children) {
						if(z->layer_set.contains(l)){
							lcl = z;
							break;
						}
					}
				}
				while(lcc!=c){
					lcc->layer_set.reset(x);
					lcc->layer_set.set(l);
					lcc->stage.clear();
					lcc = lcc->children.front();
				}
				size_t i = find(c->parent->children, c);
				size_t j = find(lnode->parent->children, lnode);
				c->parent->children[i] = lnode;
				lnode->parent->children[j] = c;
				swap(lnode->parent,c->parent);
				swap(lnode->num_batch,c->num_batch);
				//std::cout << (int)x << ' ' << (int)l <<std::endl;
				// Now we'll reset the whole seg.
				if(back == root){
					back->children[k]->isNewNode = true;
					back->children[k+1]->isNewNode = true;
				}else{
					while(back->parent->parent) back = back->parent;
					back->isNewNode = true;
				}
				ok=true;
			}break;
			case 2:{
				// Delete parent, merge to grandma.
				LTreeNode* par = lnode->parent;
				if(par == nullptr) break;
				LTreeNode* grandma = par->parent;
				if(grandma == nullptr) break;
				size_t i = find(grandma->children, par);
				grandma->children.erase(grandma->children.begin()+i);
				grandma->children.insert(grandma->children.begin()+i, par->children.begin(), par->children.end());
				grandma->stage.clear();
				for(auto x : par->children){
					x->parent = grandma;
					x->t = LTreeNode::NodeType::L;
					x->num_batch = par->num_batch;
				}
				// Now we'll reset the whole seg.
				if(grandma == root){
					for(auto x : par->children){
						x->isNewNode = true;
					}
				}else{
					while(grandma->parent->parent) grandma = grandma->parent;
					grandma->isNewNode = true;
				}
				par->children.clear();
				delete par;
				ok=true;
			}break;
			case 3:{
				// Add new parent, select a range of childs.
				LTreeNode* par = lnode->parent;
				if(par == nullptr || par->children.size() <= 2) break;
				size_t i = find(par->children, lnode);
				size_t x= par->children.size()-1;
				size_t j;
				do{
					j=randInt(x);
					if(j>=i)++j;
				}while(i*j==0&&i+j==x);
				x = MIN(i,j);
				j=MAX(i,j)+1;
				i=x;
				// Check depth limit
				if(par->height + depth > max_depth){
					bool p = false;
					for(size_t x=i;x<j;++x){
						if(par->children[x]->height + depth >= max_depth){
							p=true;
							break;
						}
					}
					if(p) break;
				}
				par->stage.clear();
				LTreeNode* new_par;
				bool T_under_T = false;
				if((par->parent == nullptr) && (par->t == LTreeNode::NodeType::T)){
					switch(sa_type){
						case 0: T_under_T = withProb(0.5); break;
						case 1: break;
						case 2: T_under_T = true; break;
						default: assert(false);
					}
				}
				new_par=new LTreeNode(Bitset(),lnode->num_batch,nullptr, T_under_T ? LTreeNode::NodeType::T : LTreeNode::NodeType::L);
				new_par->children.insert(new_par->children.begin(), par->children.begin()+i, par->children.begin()+j);
				new_par->parent = par;
				par->children.erase(par->children.begin()+i, par->children.begin()+j);
				par->children.insert(par->children.begin()+i, new_par);
				for(auto x:new_par->children){
					x->parent = new_par;
					if(T_under_T) x->t = LTreeNode::NodeType::L;
					else if(par->t == LTreeNode::NodeType::T) flat_bat(x);
				}
				// Now we'll reset the whole seg.
				while(new_par->parent->parent) new_par = new_par->parent;
				new_par->isNewNode = true;
				ok=true;
			}break;
			case 4:{
				// Put batch down
				LTreeNode* cur = lnode->parent;
				int d = randInt(depth);
				while(d-->0){
					cur = cur->parent;
				}
				if(cur->children.front()->num_batch == cur->num_batch){
					break;
				}
				for(auto x:cur->children){
					x->num_batch *= 2;
				}
				// Now we'll reset the whole seg.
				if(cur == root){
					cur->isNewNode = true;
				}else{
					while(cur->parent->parent) cur = cur->parent;
					cur->isNewNode = true;
				}
				ok=true;
			}break;
			case 5:{
				// Put batch up
				LTreeNode* cur = lnode->parent;
				int d = randInt(depth);
				while(d-->0){
					cur = cur->parent;
				}
				if(cur->children.front()->num_batch == 1){
					break;
				}
				for(auto x:cur->children){
					halv_bat(x);
				}
				// Now we'll reset the whole seg.
				if(cur == root){
					cur->isNewNode = true;
				}else{
					while(cur->parent->parent) cur = cur->parent;
					cur->isNewNode = true;
				}
				ok=true;
			}break;
			case 6:{
				// Put lnode into the cut before/after it (or out of its parent).
				LTreeNode* par = lnode->parent;
				if(par->children.size() <= 2) break;
				size_t node_pos = find(par->children, lnode);
				bool put_before = (randInt(2) == 0);
				if(put_before?(node_pos > 0):(node_pos < par->children.size()-1)){
					size_t next_pos = node_pos + (put_before?-1:1);
					LTreeNode* cut = par->children[next_pos];
					if(cut->t == LTreeNode::NodeType::L) break;
					// Put lnode under cut.
					par->stage.clear();
					cut->stage.clear();
					par->children.erase(par->children.begin()+node_pos);
					if(put_before){
						cut->children.push_back(lnode);
					}else{
						cut->children.insert(cut->children.begin(), lnode);
					}
					lnode->parent = cut;

					lnode->num_batch = cut->get_bgrp_size();
					cut->layer_set.set(l);
					// Now we'll reset the whole seg.
					assert(cut != root);
					while(cut->parent->parent) cut = cut->parent;
					cut->isNewNode = true;
				}else{
					LTreeNode* grandma = par->parent;
					if(grandma == nullptr) break;
					// Put lnode under par->parent.
					size_t par_pos = find(grandma->children, par);
					size_t insert_pos = par_pos + (put_before?0:1);
					par->stage.clear();
					grandma->stage.clear();
					par->children.erase(par->children.begin()+node_pos);
					grandma->children.insert(grandma->children.begin()+insert_pos, lnode);
					lnode->parent = grandma;

					lnode->num_batch = grandma->get_bgrp_size();
					par->layer_set.reset(l);
					// Now we'll reset the whole seg.
					if(grandma == root){
						lnode->isNewNode = true;
						par->isNewNode = true;
					}else{
						while(grandma->parent->parent) grandma = grandma->parent;
						grandma->isNewNode = true;
					}
				}
				ok=true;
			}break;
			default:
				break;
			}
			++cur_tries;
			++num_tries;
			if(cur_tries >= print_tries){
				out << "[Warning] After " << cur_tries << " tries." << endl;
				print_tries *= 2;
			}
		}while(!ok);
		//std::cout << t << std::endl;
		if(op_type) *op_type = t;
		root->init_root();
		return root;
	}

	bool sa_accept(cost_t cur_cost, cost_t new_cost, int round){
		if(new_cost <= cur_cost) return true;
		/*
		 * T(x) = a+c/(b+x)
		 * a + c/b = 0.1
		 * a + c/(b+0.5) = 0.01
		 * a + c/(b+1) = 0
		 * a = -1 / 80
		 * b = 0.01*0.5/(0.1*0.5-0.01) = 0.125
		 * c = 9 / 640
		 * T(x) = 1/10 * (1-x)/(1+8x)
		 */
		double x = round;
		x /= nrounds;
		// Since only 1/100 are good, multiply T by 0.7:
		double T = 0.07 * (1-x)/(1+8*x);
		double prob = exp(-((new_cost - cur_cost)/cur_cost)/T);
		//cout << x << ' ' << T << ' ' << prob << std::endl;
		return withProb(prob);
	}

	/*
	static mutex m;
	void ping_func(volatile bool& stop){
		while(true){
			for(int i=0;i<120;++i){
				this_thread::sleep_for(chrono::milliseconds(500));
				if(stop) return;
			}
			unique_lock<mutex> l(m);
			cout << "[Ping] " << cur_round << ' ' << cur_tries << ' ' << num_tries << endl;
			l.unlock();
		}
	}
	*/

	void SA_search(WholeSch& w_sch, const Cluster& c, lid_t max_depth=0, int sa_type=0){
		time_t start_time = time(nullptr);
		int nvalid = 0, naccept = 0;
		LTreeNode*& min_node = w_sch.tree;
		SchNode*& min_res = w_sch.sch;
		LTreeNode* cur_node = w_sch.tree;
		SchNode* cur_res = w_sch.sch;
		int op_type;
		map<int, int> accept_num;
		map<int, int> valid_num;
		for(int i=0;i<NUM_OP;++i){
			accept_num[i] = 0;
			valid_num[i] = 0;
		}
		bool using_best = false;

		bool valid_op[NUM_OP];
		for(int i=0;i<NUM_OP;++i) valid_op[i] = true;
		if(network->is_chain()) valid_op[0] = valid_op[1] = false;
		if(cur_node->get_tot_batch() == 1) valid_op[4] = valid_op[5] = false;
		num_tries = 0;
		cur_tries = 0;
		int print_intv = nrounds/30;
		cur_round=0;
		// bool stop_ping = false;
		// std::thread ping(ping_func, ref(stop_ping));
		for(;cur_round<nrounds;++cur_round){
			if((cur_round+1)%print_intv == 0){
				//unique_lock<mutex> l(m);
				out << cur_round << ' ' << cur_res->get_cost().cost() << ' ' << (num_tries * 1.0) /print_intv << endl;
				//l.unlock();
				num_tries = 0;
			}
			if(cur_round >= 0.90*nrounds && !using_best){
				using_best = true;
				if(cur_node != min_node){
					//unique_lock<mutex> l(m);
					out << "Switch to best solution." << endl;
					//l.unlock();
					delete cur_node;
					delete cur_res;
					cur_node = min_node;
					cur_res = min_res;
				}
			}
			LTreeNode* new_tree = sa_change(cur_node, valid_op, max_depth, sa_type, &op_type);
			SchNode* new_res;
			if(new_tree->isNew()){
				new_res = Cut::newNode(new_tree, c, nullptr);
			}else{
				new_res = cur_res->copy();
				assert(new_tree->isModified());
				new_res->searchInc(new_tree,false);
			}
			if(!new_res->is_valid()){
				delete new_tree;
				delete new_res;
				continue;
			}
			new_tree->confirm();
			++nvalid;
			++valid_num[op_type];
			cost_t new_cost = new_res->get_cost().cost();
			if(new_cost < min_res->get_cost().cost()){
				if(cur_node != min_node){
					delete min_node;
					delete min_res;
				}
				min_node = new_tree;
				min_res = new_res;
			}

			if(sa_accept(cur_res->get_cost().cost(), new_cost, cur_round)){
				//cout << "accept!" << endl;
				if(cur_node != min_node){
					delete cur_node;
					delete cur_res;
				}
				cur_node = new_tree;
				cur_res = new_res;
				++naccept;
				++accept_num[op_type];
			}else{
				if(new_tree != min_node){
					delete new_tree;
					delete new_res;
				}
			}
		}
		if(cur_node != min_node){
			delete cur_node;
			delete cur_res;
		}
		time_t end_time = time(nullptr);
		//stop_ping = true;
		//ping.join();
		out << "Elapsed: " << end_time - start_time << "s ";
		out << "Valid: " << nvalid << " (" << (nvalid*100.0)/nrounds << "%) ";
		out << "Accept: " << naccept << " (" << (naccept*100.0)/nrounds << "%)" << endl;
		out << "Per OP: ";
		for(int i=0;i<NUM_OP;++i){
			if(i>0) out << ", ";
			out << accept_num[i] << '/' << valid_num[i];
		}
		out<<endl;
	}
};

int SAEngine::nrounds;

void LP_search(lid_t num_layer, len_t tot_batch, Cluster& c, WholeSch& w_sch, bool has_S, bool has_T){
	if(!has_S && !has_T){
		throw std::invalid_argument("Either has_S or has_T must be true.");
	}
	LTreeNode*& tree_node = w_sch.tree;
	SchNode*& sch_res = w_sch.sch;
	SchNode::SchCost LP_cost;
	LTreeNode** LP_DP = new LTreeNode*[num_layer];
	SchNode* tmp = nullptr;
	SchNode* last = nullptr;
	LTreeNode* root_Node;
	for(int i=0;i<num_layer;++i) {
		LP_DP[i]=nullptr;
		LP_cost.energy = energy_inf;
		// cout << "\tStart " << network->getNode(i).name() << endl;
		for(int j=-1;j<i;++j){
			if(j != -1 && LP_DP[j] == nullptr) continue;
			for(len_t l_bat=1;l_bat<=4&&l_bat<=tot_batch;l_bat*=2){
				for(int last_T = has_S?0:1; last_T < (has_T?2:1); ++last_T){
					if(j==i-1 && has_S && has_T && last_T > 0) break;
					if(j==-1){
						// Create top T-cut.
						root_Node = new LTreeNode(Bitset(), tot_batch, nullptr, LTreeNode::NodeType::T);
					}else {
						// Use top T-cut from DP.
						root_Node = LP_DP[j]->copy();
						root_Node->reset_lset();
					}
					if(j == i-1){
						// New LNode: last layer.
						(void) new LTreeNode(i, tot_batch, root_Node);
					}else{
						LTreeNode* cur_Cut = new LTreeNode(Bitset(), tot_batch, root_Node, (last_T == 0) ? LTreeNode::NodeType::S : LTreeNode::NodeType::T);
						for(int k=j+1;k<=i;++k){
							(void) new LTreeNode(k, l_bat, cur_Cut);
						}
					}
					root_Node->init_root();
					// TODO: change to inc. search
					tmp = Cut::newNode(root_Node, c, nullptr);
					if(!tmp->is_valid()){
						delete tmp;
						delete root_Node;
						continue;
					}
					root_Node->confirm();
					if(tmp->get_cost().cost() >= LP_cost.cost()){
						delete tmp;
						delete root_Node;
						continue;
					}
					if(LP_DP[i]) delete LP_DP[i];
					LP_DP[i] = root_Node;
					LP_cost = tmp->get_cost();
					if(last) delete last;
					last = tmp;
				}
			}
		}
	}
	tree_node = LP_DP[num_layer-1];
	if(tree_node == nullptr){
		std::cout << "Warning: no scheme found in LP_search!" << std::endl;
		sch_res = nullptr;
		if(last) delete last;
	}else{
		sch_res = last;
	}
	for(int i=0; i<num_layer-1; ++i) if(LP_DP[i]) delete LP_DP[i];
	delete[] LP_DP;
	return;
}

std::vector<double> buffer(int width, vol_t size) {
	vector<double> access;//per byte 0 is read, 1 is write
	access.resize(2);
	vol_t size_;
	if (width >= 512) {
		size_ = size / (width / 256);
		access = buffer(256, size_);
		access[0] *= sqrt(size / size_);
		access[1] *= sqrt(size / size_);
		return access;
	}
	if (width == 256) {
		if (size == 32 KB) {
			access[0] = 0.065678125;
			access[1] = 0.0641375;
		}
		else if (size == 16 KB) {
			access[0] = 0.06311875;
			access[1] = 0.05563125;
		}
		else if (size == 8 KB) {
			access[0] = 0.051290625;
			access[1] = 0.04560625 ;
		}
		else if (size == 4 KB) {
			access[0] = 0.049;
			access[1] = 0.064;
		}
		else {
			throw runtime_error("Cannot find buffer energy.");
		}
	}
	if (width == 128) {
		if (size == 32 KB) {
			access[0] = 0.106675;
			access[1] = 0.10539375;
		}
		else if (size == 16 KB) {
			access[0] = 0.06848125;
			access[1] = 0.06684375;
		}
		else if (size == 8 KB) {
			access[0] = 0.06553125;
			access[1] = 0.057975;
		}
		else if (size == 4 KB) {
			access[0] = 0.05336875;
			access[1] = 0.0476625;
		}
		else if (size == 2 KB) {
			access[0] = 0.06025625;
			access[1] = 0.06025625;
		}
		else {
			throw runtime_error("Cannot find buffer energy.");
		}
	}
	if (width == 64) {
		if (size == 32 KB) {
			access[0] = 0.194775;
			access[1] = 0.1923;
		}
		else if (size == 16 KB) {
			access[0] = 0.112;
			access[1] = 0.110675;
		}
		else if (size == 8 KB) {
			access[0] = 0.0740875;
			access[1] = 0.0722625;
		}
		else if (size == 4 KB||size == 4.5 KB) {
			access[0] = 0.0703625;
			access[1] = 0.0626625;
		}
		else if (size == 2 KB) {
			access[0] = 0.057525;
			access[1] = 0.051775;
		}
		else {
			throw runtime_error("Cannot find buffer energy.");
		}
	}
	if (width == 32) {
		if (size == 32 KB) {
			access[0] = 0.194775;
			access[1] = 0.1923;
		}
		else if (size == 16 KB) {
			access[0] = 0.112;
			access[1] = 0.110675;
		}
		else if (size == 8 KB) {
			access[0] = 0.11608;
			access[1] = 0.13898;
		}
		else if (size == 4 KB) {
			access[0] = 0.07422;
			access[1] = 0.09414;
		}
		else if (size == 2 KB) {
			access[0] = 0.06326;
			access[1] = 0.08034;
		}
		else if (size == 1 KB) {
			access[0] = 0.07952;
			access[1] = 0.0981;
		}
		else {
			throw runtime_error("Cannot find buffer energy.");
		}
	}
	return access;
}
double wafer_util(double area) {
	if (	 area <= 25000000) {
		return 0.865;
	}
	else if (area <= 50000000) {
		return 0.859;
	}
	else if (area <= 100000000) {
		return 0.849;
	}
	else if (area <= 200000000) {
		return 0.815;
	}
	else if (area <= 300000000) {
		return 0.806;
	}
	else if (area <= 400c) {
		return 0.792;
	}
	else if (area <= 500000000) {
		return 0.778;
	}
	else if (area <= 600000000) {
		return 0.764;
	}
	else if (area <= 700000000) {
		return 0.743;
	}
	else if (area <= 800000000) {
		return 0.743;
	}
}
int main(int argc, char** argv){
	unsigned int seed = time(NULL);
	srand(seed);
	cout << "Seed: " << seed << " ";
	std::cout.precision(4);

	// Get input parameters.
	int mm, nn, xx, yy, ss, bb, rr, ff, xcut, ycut, _serdes_lane, _DRAM_bw, _NoC_bw, _mac_dim, _ul3, total_tops;
	cin >> mm >> nn >> xx >> yy >> ss >> bb >> rr >> ff >> xcut >> ycut >> _serdes_lane >> _DRAM_bw >> _NoC_bw >> _mac_dim >> _ul3 >> total_tops;
	// assert(total_tops == 2*xx*yy*_mac_dim*_mac_dim);
	//mm means core_microarch: 0 means PolarCore (NVDLA-STYLE), which is recommenmanded; 1 means Eyeriss Core.
	//nn means network choice
	//xx & yy means the number of cores in the x and y axis, respectively. Default noc topology is mesh.
	//ss means stride, whose definition can be found in ISCA2023 SET, HPCA2024GEMINI, ASPLOS2019TANGRAM.
	//bb means batch size (2^n is recommanded).
	//rr means exploration rounds number, which will be multiplied with a coefficient.
	//ff means optimization goal, concrete formula can be found in this main.cpp.
	//xcut & ycut means the chiplet partition granularity at the x and y dimension.
	//the remaining few parameters represent the architecture parameters as their names.
	/********************* INPUT *********************/
	bw_t serdes_lane = _serdes_lane;
	NoC::DRAM_bw = _DRAM_bw/1024/4;     
	SchNode::DRAM_bw = _DRAM_bw/1024/4;   
	double DRAM_bw_each = _DRAM_bw / 1024 / 4;
	NoC::NoC_bw = _NoC_bw;     
	
	//core & chiplet num
	Cluster::xlen = xx;     
	Cluster::ylen = yy;     
	NoC::x_cut = xcut;   
	NoC::y_cut = ycut;
	NoC::soc = xcut * ycut == 1;
	std::uint16_t mac_dim = _mac_dim;  
	vol_t ul3_ = _ul3 KB;            

	NoC::NoP_bw = serdes_lane;
	NoC::x_step = Cluster::xlen / NoC::x_cut;
	NoC::y_step = Cluster::ylen / NoC::y_cut;
	if (xx % ss != 0 || xx % NoC::x_cut != 0 || yy % NoC::y_cut != 0) {
		throw runtime_error("Chiplet partition is invalid.");
	}

	//DENSITY
	density_t SRAM_den = 0.0964 * 8;
	density_t MAC_den = 57.9;//8-bit
	density_t LR_mac_den = MAC_den * 8;
	density_t NoP_len = 333; 
	density_t NoP_wid = 800;
	density_t NoC_den = 16781.312;
	density_t DDR_PHY_den = 6.53 * 1000000;
	density_t DDR_ctrl_den = 510 * 1000 * 0.09;
	double yield = 0.9;

	std::uint16_t vector_len;
	std::uint16_t lane_len;
	if (mac_dim == 512) {
		vector_len = 4;
		lane_len = 8;
	}
	else if (mac_dim == 1024) {
		vector_len = 8;
		lane_len = 8;
	}
	else if (mac_dim == 2048) {
		vector_len = 8;
		lane_len = 16;
	}
	else if (mac_dim == 4096) {
		vector_len = 16;
		lane_len = 16;
	}
	else if (mac_dim == 8192) {
		vector_len = 16;
		lane_len = 32;
	}
	else {
		throw runtime_error("MAC scale not supported·");
	}
	NoC::seperate_IO = true;
	double turnover_factor = 0.3/0.5;
	ofm_ubuf_vol = 10 KB;
	NoC::NoP_hop_cost = 2 * 8; //DSE->3.3*8 ; Simba->2*8
	NoC::NoC_hop_cost = 0.8 * 8;
	NoC::DRAM_acc_cost = 10.5 * 8;
	energy_t LR_mac_cost = 0.0873; //IEEE FP16
	Core::numMac_t LR_mac_num = vector_len * lane_len *16 / 16;
	PolarCore::Buffer al1, wl1, ol1, al2, wl2, ol2, ul3;
	PolarCore::PESetting s(vector_len, lane_len, 0.018);
	PolarCore::Bus bus(4, 4, 0.018, 64);
	al1.Size = 8 * vector_len / 8 KB; 
	ol1.Size = 2 * lane_len / 8 KB; 
	wl1.Size = 4 * lane_len*vector_len / 64 KB; 
	ol2.Size = 28 * vector_len * lane_len / 64 KB; 
	wl2.Size = 0;//256 KB;
	ul3.Size = ul3_; //16 64-bit IO 64KB 1-port MBSRAM
	SchNode::ubuf = ul3.Size;

	double tops = 16*vector_len*lane_len * Cluster::xlen * Cluster::ylen * 2;
	tops /= 1024;
	//NUMBER
	double core_num = Cluster::xlen * Cluster::ylen;
	double compute_die_num = NoC::x_cut * NoC::y_cut;
	double NoP_side_num;//how many sides of 
	if (compute_die_num == 1) {
		NoP_side_num = 0;
	}
	else if (compute_die_num == 2) {
		NoP_side_num = 2;
	}
	else {
		NoP_side_num = 4;
	}

	//COST
	double cost_silicon_mm_compute = 0.084887 ;
	double cost_silicon_mm_IO = 0.056383;
	double cost_os = 0.005; //per mm^2
	double os_area_scale_factor = 4;// os area will be larger than chip
	double ddr_cost = 3.5;//per 16bithttps://www.dramexchange.com/
	double os_ocst_factor;//larger os will be more expensive 
	double post_layout_scale = 2.2;
	double control_unit_prop = 1.05;
	double DFT_prop = 1.05;
	//*********************DIE AREA*********************
	double sram_area_per_core = (ul3_ + (al1.Size + wl1.Size + ol1.Size)*16) * SRAM_den;
	double mac_area_per_core = 16*vector_len * lane_len * MAC_den + LR_mac_num*LR_mac_den;
	double NoC_area_per_core = NoC::NoC_bw * NoC_den;
	double core_area = (sram_area_per_core + mac_area_per_core + NoC_area_per_core)*post_layout_scale;
	double core_len = sqrt(core_area);
	double NoP_len_per_core = NoC::NoP_bw / 4 * NoP_len;
	double PCIe_area = 3000000 * DRAM_bw_each * 4 / 128;
	double IO_die_area = DRAM_bw_each * 4 / 44.0 * (DDR_PHY_den+DDR_ctrl_den)+ PCIe_area +(compute_die_num==1?0:(NoP_len*NoP_wid* NoC::NoP_bw / 4 *Cluster::ylen*2));//neglect other IOs; 44GB/s represents the bandwidth of a GDDR6X channel.
	double compute_die_area = 0;
	double core_len_real = core_len;
	double D2D_area_per_compute=0;
	if (compute_die_num == 1) {
		D2D_area_per_compute = 0;
	}
	else if (compute_die_num == 2) {
		D2D_area_per_compute = 1 * NoC::x_step * NoP_len_per_core * NoP_wid + 1 * NoC::y_step * NoP_len_per_core * NoP_wid;
	}
	else {
		D2D_area_per_compute = 2 * NoC::x_step * NoP_len_per_core * NoP_wid + 2 * NoC::y_step * NoP_len_per_core * NoP_wid;
	}
	compute_die_area = core_len_real * NoC::x_step * core_len_real * NoC::y_step + D2D_area_per_compute;
	compute_die_area *= (control_unit_prop * DFT_prop);
	IO_die_area *= (control_unit_prop * DFT_prop);
	double total_die_area = compute_die_area * compute_die_num + IO_die_area;
	double os_area = os_area_scale_factor * total_die_area;

	// Check if areas are too large.
	if (compute_die_num == 1) {
		if (total_die_area > 858*1000000) {
			cout << "`total_die_area` too large." << endl;
			return 1;
		}
	}else{
		if(IO_die_area>858*1000000 || compute_die_area>858*1000000) {
			cout << "`IO_die_area` or `compute_die_area` too large." << endl;
			return 1;
		}
	}
	//*********************DIE AREA*********************
	if (NoC::x_cut * NoC::y_cut == 1) {
		os_ocst_factor = 1;
	}
	else if (os_area <= 30 * 30 * 1000000) {
		os_ocst_factor = 1.5;
	}
	else if (os_area <= 55 * 55 * 1000000) {
		os_ocst_factor = 2;
	}
	else {
		os_ocst_factor = 4;
	}
	//*********************COST CALC*********************
	double cost_overall = 0;
	double yield_compute_die = 0;
	double yield_IO_die = 0;
	double cost_compute = 0;
	double cost_IO = 0;
	double cost_os_overall = 0;
	double yield_soc = 0;
	double cost_soc = 0;
	if (compute_die_num != 1) {
		yield_compute_die = pow(yield, compute_die_area / 40000000);
		yield_IO_die = pow(yield, IO_die_area / 4 / 40000000);
		cost_compute = compute_die_area * compute_die_num / yield_compute_die / 1000000 * cost_silicon_mm_compute/wafer_util(compute_die_area);
		cost_IO = IO_die_area / 1000000 * cost_silicon_mm_IO/ wafer_util(IO_die_area/4)/yield_IO_die + DRAM_bw_each * 4 / 44.0 * ddr_cost;
		cost_os_overall = os_area * os_ocst_factor / 1000000 * cost_os;
		cost_overall = cost_compute + cost_IO + cost_os_overall;
	}
	else {
		yield_soc = pow(yield, total_die_area / 40000000);
		cost_soc = total_die_area / yield_soc / 1000000 * cost_silicon_mm_compute/ wafer_util(total_die_area) + DRAM_bw_each * 4 / 44 * ddr_cost;
		cost_os_overall = os_area * os_ocst_factor / 1000000 * cost_os;
		cost_compute = cost_soc;
		cost_IO = 0;
		cost_overall = cost_soc + cost_os_overall;
	}
	//*********************COST CALC*********************

	
	
	al2.Size = 0;
	al1.RCost = (buffer(vector_len * 8, al1.Size)[0]+0.1*lane_len/8) * 8 * turnover_factor;
	al1.WCost = buffer(vector_len * 8, al1.Size)[1] * 8 * turnover_factor ;
	wl1.RCost = buffer(vector_len * 8, wl1.Size)[0] * 8 * turnover_factor ;
	wl1.WCost = buffer(vector_len * 8, wl1.Size)[1] * 8 * turnover_factor ;
	ol1.RCost = buffer(lane_len * 16, ol1.Size)[0] * 8 * turnover_factor ;
	ol1.WCost = buffer(lane_len * 16, ol1.Size)[1] * 8 * turnover_factor ;
	ol2.RCost = 0.07648125 * 8 * ul3_ / (1024 KB) * turnover_factor ;// ol2 is a small part of ul3
	ol2.WCost = 0.0989875 * 8 * ul3_ / (1024 KB) *turnover_factor ;
	ul3.RCost = 0.217125 * 8 * ul3_ /(1024 KB) * turnover_factor; 
	ul3.WCost = 0.234025 * 8 * ul3_ /(1024 KB)* turnover_factor;  
	al2.RCost = al2.WCost = 0;
	wl2.RCost = wl2.WCost = 0;
	
	PolarCore core(s,bus,al1,wl1,ol1,al2,wl2,ol2,ul3,LR_mac_num,LR_mac_cost);
	PolarMapper mapper(core);

	EyerissCore::Buffer _al1, _wl1, pl1, ul2;
	EyerissCore::PESetting s2(mac_dim, mac_dim, 0.018);
	EyerissCore::Bus ibus(0.018, 64);
	EyerissCore::Bus wbus(0.018, 64);
	EyerissCore::Bus pbus(0.018, 64); // ifmap RC, weight RCK, psum RK

	_al1.Size = 32;
	pl1.Size = 1;
	_wl1.Size = 128;
	ul2.Size = 1024 KB;

	_al1.RCost = 0.0509 * 8 * turnover_factor; //8bit IO single port
	_al1.WCost = 0.0506 * 8 * turnover_factor;//0.045;
	_wl1.RCost = 0.0545 * 8 * turnover_factor; //Using 2 banks of 64
	_wl1.WCost = 0.054 * 8 * turnover_factor;//0.090;
	pl1.RCost = pl1.WCost = 0.0 * turnover_factor;
	ul2.RCost = 0.1317125 * 8 * turnover_factor;
	ul2.WCost = 0.234025 * 8 * turnover_factor;

	EyerissCore core2(s2, ibus, wbus, pbus, _al1, _wl1, pl1, ul2, LR_mac_num, LR_mac_cost);
	EyerissMapper mapper2(core2);

	CoreMapper* cMapper;
	if(mm == 0){
		cMapper = &mapper;
	}else{
		cMapper = &mapper2;
	}
	// TOPS
	

	// 2 GB/TOPS
	if (NoC::DRAM_bw == 0) {
		NoC::DRAM_bw = 0.75 * (tops / 4);
		SchNode::DRAM_bw = 0.75 * (tops / 4);
		NoC::NoC_bw = 4;//NoC::DRAM_bw / 4;
	}
	NoC::interleave = false;

	StdLayerEngine engine(cMapper);
	FastLayerEngine engine_fast(cMapper);
	SchNode::layerMapper = &engine;
	SchNode::layerMapper_fast = &engine_fast;
	Light_placement::fast_buffer = &engine_fast;

	std::string net_name;
	switch (nn) {
	case 0:
		network = &darknet19;
		net_name="darknet19";
		break;
	case 1:
		network = &vgg19;
		net_name="vgg";
		break;
	case 2:
		network = &resnet50;
		net_name="resnet";
		break;
	case 3:
		network = &googlenet;
		net_name="goog";
		break;
	case 4:
		network = &resnet101;
		net_name="resnet101";
		break;
	case 5:
		network = &densenet;
		net_name="densenet";
		break;
	case 6:
		network = &inception_resnet_v1;
		net_name="ires";
		break;
	case 7:
		network = &gnmt;
		net_name="gnmt";
		break;
	case 8:
		network = &lstm;
		net_name="lstm";
		break;
	case 9:
		network = &zfnet;
		net_name="zfnet";
		break;
	case 10:
		network = &transformer;
		net_name="trans";
		break;
	case 11:
		network = &transformer_cell;
		net_name="trans_cell";
		break;
	case 12:
		network = &PNASNet;
		net_name="pnas";
		break;
	case 13:
		network = &resnext50;
		net_name="resnext50";
		break;
	case 14:
		network = &resnet152;
		net_name="resnet152";
		break;
	// TODO: Support more DNNs.
	default:
		throw runtime_error("Model not supported.");
	}

	
	len_t tot_batch = bb;
	SchNode::tot_batch = tot_batch;

	Cluster::stride = ss;
	Cluster::min_util = 0.75;

	//SchNode::tot_batch = 64;
	lid_t num_layer = network->len();
	SAEngine::nrounds = rr;
	network->set_utime(*cMapper);
	
	Cluster c(0,Cluster::xlen * Cluster::ylen);

	
	
	//NoP link caculation
	if (compute_die_num != 1) {
		for (mlen_t i = 0; i < Cluster::ylen; i++) {
			NoC::NoP_links.insert((- 1* Cluster::ylen + i) * 4);
			NoC::NoP_links.insert((0 * Cluster::ylen + i) * 4+2);
			NoC::NoP_links.insert(((Cluster::xlen-1)* Cluster::ylen + i) * 4);
			NoC::NoP_links.insert((Cluster::xlen * Cluster::ylen + i) * 4 + 2);
			for (mlen_t j = Cluster::xlen / NoC::x_cut; j < Cluster::xlen;) {
				NoC::NoP_links.insert(((j - 1) * Cluster::ylen + i) * 4);
				NoC::NoP_links.insert((j * Cluster::ylen + i) * 4 + 2);
				j += Cluster::xlen / NoC::x_cut;

			}
		}

		for (mlen_t i = 0; i < Cluster::xlen; i++) {
			for (mlen_t j = Cluster::ylen / NoC::y_cut; j < Cluster::ylen;) {
				NoC::NoP_links.insert((i * Cluster::ylen + j - 1) * 4 + 3);
				NoC::NoP_links.insert((i * Cluster::ylen + j) * 4 + 1);
				j += Cluster::ylen / NoC::y_cut;
			}
		}
	}


	NoC::DRAM_num = 4;
	NoC::DRAM_router_num = Cluster::ylen / 2;
	NoC::dram_list.resize(NoC::DRAM_num);
	for (mlen_t y = 0; y < NoC::DRAM_num; ++y) {
		NoC::dram_list[y].resize(NoC::DRAM_router_num);
	}

	for (mlen_t y = 0; y < Cluster::ylen; ++y) {
		if (y < NoC::DRAM_router_num) {
			NoC::dram_list[0][y] = {-1, y};
			NoC::dram_list[2][y] = {static_cast<mlen_t>(Cluster::xlen), y};
		}
		else if(y>= Cluster::ylen - NoC::DRAM_router_num){
			NoC::dram_list[1][y - Cluster::ylen / 2] = { -1, y };
			NoC::dram_list[3][y - Cluster::ylen / 2] = { static_cast<mlen_t>(Cluster::xlen), y };
		}
	}
	
	NoC::dram_list_base.resize(2*Cluster::ylen);
	for (mlen_t y = 0; y < Cluster::ylen; ++y) {
		NoC::dram_list_base[y] = { -1, y};
		NoC::dram_list_base[Cluster::ylen + y] = { static_cast<mlen_t>(Cluster::xlen), y };
	}

	switch (ff){
		case 1:
			// This is the default cost_func.
			cost_func = [](energy_t e, cycle_t t){return e*t;};
			break;
		case 0:
			cost_func = [](energy_t, cycle_t t){return t;};
			break;
		case -1:
			cost_func = [](energy_t e, cycle_t){return e;};
			break;
		default:
			if(ff > 0){
				cost_func = [=](energy_t e, cycle_t t)->cost_t{return pow(e,ff)*t;};
			}else{
				cost_func = [=](energy_t e, cycle_t t)->cost_t{return e*pow(t,-ff);};
			}
	}

/* 
	`stschedule` tool prints input parameters, search time, tree info and cost info.
	These output info are directly writed to the `stdout`. Redirect `stdout` if needed.
 */	
	cout << "Mapper " << ((cMapper == &mapper)?"polar":"eyeriss");
	cout << " Network " << net_name;
	cout << " Mesh " << (int)Cluster::xlen << '*' << (int)Cluster::ylen;
	cout << " Batch " << tot_batch << endl;
	cout << "compute_die_area: " << compute_die_area << endl;
	cout << "IO_die_area: " << IO_die_area << endl;
	cout << "os_area: " << os_area << endl;
	cout << "cost_compute: " << cost_compute << endl;
	cout << "cost_IO: " << cost_IO << endl;
	cout << "cost_os_overall: " << cost_os_overall << endl;
	cout << "cost_soc: " << cost_soc << endl;

#if FORMAT_OUTPUT
	printf("INPUT_PARAMETERS: %1d %2d %3d %3d %3d %3d %3d %2d %3d %3d %3d %8d %3d %4d %4d %8d\n", mm, nn, xx, yy, ss, bb, rr, ff, xcut, ycut, _serdes_lane, _DRAM_bw, _NoC_bw, _mac_dim, _ul3, total_tops);
#else
	cout << "INPUT_PARAMETERS: " << mm << ' ' << nn << ' ' << xx << ' ' << yy << ' ' 
			<< ss << ' ' << bb << ' ' << rr << ' ' << ff << ' ' 
			<< xcut << ' ' << ycut << ' ' << _serdes_lane << ' ' 
			<< _DRAM_bw << ' ' << _NoC_bw << ' ' << _mac_dim << ' ' 
			<< _ul3 << ' ' << total_tops << endl;
#endif
	Segment_scheme *scheme;

#if FORMAT_OUTPUT
	FILE *file;
	file = argc > 1 ? fopen(argv[1], "a") : fopen("./temp_points.txt", "a");
	auto output = [&](int search_index) {
		fprintf(file, "%1d %2d %3d %3d %3d %3d %3d %2d %3d %3d %3d %8d %3d %4d %4d %8d %18.5f %20.5f %10lu %28.5f %32.5f %1d %20.2f %20.2f %20.2f %20.2f %20.2f %20.2f %20.2f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", 
				mm, nn, xx, yy, ss, bb, rr, ff, xcut, ycut, _serdes_lane, _DRAM_bw, _NoC_bw, _mac_dim, _ul3, total_tops, cost_overall, 
				scheme->schtree_spm->get_cost().energy, scheme->schtree_spm->get_cost().time, scheme->schtree_spm->get_cost().cost(), scheme->schtree_spm->get_cost().cost() * cost_overall, search_index,
				// Energy breakdown
				SchNode::record.ubuf, SchNode::record.buf, SchNode::record.bus, SchNode::record.mac, SchNode::record.NoC_hop_cost, SchNode::record.NoP_hop_cost, SchNode::record.DRAM_cost,
				// Cost breakdown
				compute_die_area, IO_die_area, os_area, cost_compute, cost_IO, cost_os_overall, cost_soc);
	};
#else
	ofstream writer;
	if (argc > 1) {
		writer.open(argv[1], ios::app);
	} else {
		writer.open("./temp.log", ios::app);
	}
	auto output = [&](int search_index) {
		writer << mm << ' ' << nn << ' ' << xx << ' ' << yy << ' ' 
			<< ss << ' ' << bb << ' ' << rr << ' ' << ff << ' ' 
			<< xcut << ' ' << ycut << ' ' << _serdes_lane << ' ' 
			<< _DRAM_bw << ' ' << _NoC_bw << ' ' << _mac_dim << ' ' 
			<< _ul3 << ' ' << total_tops << ' ' << cost_overall << ' ' 
			<< ' ' << scheme->schtree_spm->get_cost().energy << ' ' 
			<< ' ' << scheme->schtree_spm->get_cost().time << ' '
			<< scheme->schtree_spm->get_cost().cost() << ' ' 
			<< scheme->schtree_spm->get_cost().cost() * cost_overall << ''
			<< search_index << endl;
	};
#endif

	// Gemini mapping.
	mode_control(false, false, true);
	scheme = DP_search(SAEngine::nrounds, false, false);
	if (scheme != nullptr) {
		output(3);
		delete scheme;
	}

#if FORMAT_OUTPUT
	fclose(file);
#else
	writer.close();
#endif
	cout << endl << endl << endl;
	return 0;
}
