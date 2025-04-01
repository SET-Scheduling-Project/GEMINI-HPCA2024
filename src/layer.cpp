#include "layer.h"
#include "partition.h"
#include <cassert>
#include <cstring>

Layer::Layer(const std::string& _name, const fmap_shape& _ifm_shape, const fmap_shape& _ofm_shape, const fmap_shape& _wgt_shape)
	:name(_name), ifm_shape(_ifm_shape), ofm_shape(_ofm_shape), wgt_shape(_wgt_shape), bitwidth(8){}

Layer::Layer(const std::string& _name)
	:name(_name), bitwidth(8){}

bwidth_t Layer::get_bitwidth() const{
	return bitwidth;
}

void Layer::set_bitwidth(bwidth_t width){
	bitwidth = width;
}

const std::string& Layer::get_name() const{
	return name;
}
utime_t Layer::get_utime() const{
	return unit_time;
}

void Layer::set_utime(utime_t time){
	unit_time = time;
}

const fmap_shape& Layer::tot_ifmap_shape() const{
	return ifm_shape;
}

const fmap_shape& Layer::ofmap_shape() const{
	return ofm_shape;
}

const fmap_shape& Layer::weight_shape() const{
	return wgt_shape;
}

void ConvLayer::Workload::init(){
	K = (K == 0)?C:K;
	W = (W == 0)?H:W;
	S = (S == 0)?R:S;
	sW = (sW == 0)?sH:sW;
	update_op();
	// This is wrong! What?
	// See resnet50 (conv3_0_br)
	//assert(sH<=R && sW<=S);
	assert(tot_op>0 && sH>0 && sW>0);
}

vol_t ConvLayer::Workload::ifm_size(len_t batch_size) const{
	return ((H-1)*sH + R) * ((W-1)*sW + S) * C * batch_size;
}

vol_t ConvLayer::Workload::fil_size() const{
	return R*S*C*K;
}

vol_t ConvLayer::Workload::ofm_size(len_t batch_size) const{
	return H*W*K*batch_size;
}

void ConvLayer::Workload::update_op(){
	tot_op = static_cast<access_t>(C)*K*R*S*H*W;
}

access_t ConvLayer::Workload::calc_op(len_t batch_size) const{
	return tot_op*batch_size;
}

ConvLayer::ConvLayer(const std::string& _name, const ConvLayer::Workload& _wl)
	:Layer(_name), wl(_wl){
	wl.init();
	ifm_shape = fmap_shape(wl.C, (wl.H - 1) * wl.sH + wl.R, (wl.W - 1) * wl.sW + wl.S);
	ofm_shape = fmap_shape(wl.K, wl.H, wl.W);
	wgt_shape = fmap_shape(wl.K, wl.C, wl.R*wl.S);
	wgt_size = wl.fil_size();
}

const fmap_shape& ConvLayer::real_ifmap_shape() const{
	return padded_ifm_shape;
}

vol_t ConvLayer::weight_size() const{
	return wgt_size;
}

const ConvLayer::Workload& ConvLayer::get_workload() const{
	return wl;
}

bool ConvLayer::set_padded_ifm(const fmap_shape& padded_shape){
	if(padded_shape.c != ifm_shape.c) return false;
	padded_ifm_shape.c = ifm_shape.c;
	if(padded_shape.h > ifm_shape.h){
		if(padded_shape.h > wl.H * wl.sH) return false;
		pad_h = 0;
		padded_ifm_shape.h = ifm_shape.h;
	}else{
		len_t tot_ph = ifm_shape.h - padded_shape.h;
		if(tot_ph > 2*(wl.R - 1)) return false;
		pad_h = tot_ph/2;
		padded_ifm_shape.h = padded_shape.h;
	}
	if(padded_shape.w > ifm_shape.w){
		if(padded_shape.w > wl.W * wl.sW) return false;
		pad_w = 0;
		padded_ifm_shape.w = ifm_shape.w;
	}else{
		len_t tot_pw = ifm_shape.w - padded_shape.w;
		if(tot_pw > 2*(wl.S - 1)) return false;
		pad_w = tot_pw/2;
		padded_ifm_shape.w = padded_shape.w;
	}
	padded_ifm_shape.update_size();
	return true;
}

access_t ConvLayer::get_num_op(len_t batch_size) const{
	return wl.calc_op(batch_size);
}

void ConvLayer::ofm_to_ifm(fmap_range& ofm_range) const{
	ofm_range.c = {0, wl.C};
	ofm_range.h.from = ofm_range.h.from * wl.sH;
	ofm_range.h.from = (ofm_range.h.from > pad_h)?(ofm_range.h.from - pad_h):0;
	ofm_range.w.from = ofm_range.w.from * wl.sW;
	ofm_range.w.from = (ofm_range.w.from > pad_w)?(ofm_range.w.from - pad_w):0;
	ofm_range.h.to = (ofm_range.h.to-1) * wl.sH + wl.R - pad_h;
	ofm_range.w.to = (ofm_range.w.to-1) * wl.sW + wl.S - pad_w;
	ofm_range.h.to = MIN(ofm_range.h.to, padded_ifm_shape.h);
	ofm_range.w.to = MIN(ofm_range.w.to, padded_ifm_shape.w);
}

void ConvLayer::ofm_to_wgt(fmap_range& ofm_range) const{
	// B = B(1)
	// ofm_range.b = {0, 1};
	// C = K, H = C, W = R*S
	// ofm_range.c = ofm_range.c;
	ofm_range.h = {0, wl.C};
	ofm_range.w = {0, wl.R*wl.S};
}
FetchSch ConvLayer::set_fetch(const PartSch& partSch, vol_t size, len_t B, len_t wgt_B) const {
	// Consider fetching.
	FetchSch fetch(1, 1, 1, 1, 1, 1);

	len_t perK = DIVCEIL(ofm_shape.c, partSch.K);
	len_t perB = DIVCEIL(B, partSch.B);
	len_t perH = DIVCEIL(ofm_shape.h, partSch.H);
	len_t perW = DIVCEIL(ofm_shape.w, partSch.W);

	// Can fetch HW iff sH/sW == R/S
	bool canFetchHW;
	len_t ifmH, ifmW;
	// canFetchHW = ((wl.R == wl.sH) && (wl.S == wl.sW));
	canFetchHW = true;
	ifmH = (perH - 1) * wl.sH + wl.R;
	ifmW = (perW - 1) * wl.sW + wl.S;

	vol_t ifmMax = perB * wl.C * ifmH * ifmW;
	vol_t wgtMax = wl.C * perK * wl.R * wl.S;
	if (wgt_B) wgtMax *= perB;
	assert(ifmMax + wgtMax > size);

	// If wgt_B, also fetch batch.
	if (wgt_B) {
		ifmMax /= perB;
		wgtMax /= perB;
		fetch.B = perB;
		if (ifmMax + wgtMax <= size) {
			return fetch;
		}
	}

	// Consider fetching on K.
	vol_t wgtMin = wgtMax / perK;
	if (ifmMax + wgtMin <= size) {
		len_t maxK = (size - ifmMax) / wgtMin;
		fetch.K = DIVCEIL(perK, maxK);
		return fetch;
	}

	// Considering fetching on BHW.
	vol_t ifmMin = ifmMax;
	if (!wgt_B) {
		ifmMin = ifmMin / perB;
		if (ifmMin + wgtMax <= size) {
			len_t maxB = (size - wgtMax) / ifmMin;
			fetch.B = DIVCEIL(perB, maxB);
			return fetch;
		}
		fetch.B = perB;
	}
	if (canFetchHW) {
		// For simplicity we only fetch H.
		ifmMin = (ifmMin / ifmH) * wl.R;
		if (ifmMin + wgtMax <= size) {
			len_t maxIH = (size - wgtMax) / (ifmMin / wl.R);
			len_t maxH = (maxIH - wl.R) / wl.sH + 1;
			fetch.H = DIVCEIL(perH, maxH);
			return fetch;
		}
	}

	// Fetch both.
	if (ifmMin + wgtMin > size) {
		// can't fetch both.
		fetch.clear();
		return fetch;
	}

	// For simplicity, only consider muli-fetch weight.
	fetch.K = perK;
	fetch.B = 1;
	ifmMin = ifmMax;
	if (!wgt_B) {
		ifmMin = ifmMin / perB;
		if (ifmMin + wgtMin <= size) {
			len_t maxB = (size - wgtMin) / ifmMin;
			fetch.wgtFetch = fetch.B = DIVCEIL(perB, maxB);
			// Opt: under this B, we can fetch K less times.
			len_t maxK = (size - ifmMin * DIVCEIL(perB, fetch.B)) / wgtMin;
			fetch.K = DIVCEIL(perK, maxK);
			return fetch;
		}
		fetch.wgtFetch = fetch.B = perB;
	}
	if (canFetchHW) {
		// For simplicity we only fetch H.
		ifmMin = (ifmMin / ifmH) * wl.R;
		if (ifmMin + wgtMin <= size) {
			len_t maxIH = (size - wgtMin) / (ifmMin / wl.R);
			len_t maxH = (maxIH - wl.R) / wl.sH + 1;
			fetch.H = DIVCEIL(perH, maxH);
			fetch.wgtFetch *= fetch.H;
			// Opt: under this HW, we can fetch K less times.
			len_t curIH = (DIVCEIL(perH, fetch.H) - 1) * wl.sH + wl.R;
			len_t maxK = (size - (ifmMin / wl.R) * curIH) / wgtMin;
			fetch.K = DIVCEIL(perK, maxK);
			return fetch;
		}
	}
	// Should not be here.
	assert(false);
	return fetch;
}

vol_t ConvLayer::ifm_part(fmap_range& ifm_range, const PartSch& part) const {
	len_t perB = DIVCEIL(ifm_range.b.size(), part.B);
	len_t C = ifm_range.c.size();
	len_t perIH = ifm_range.h.size();
	if (part.H != 1) {
		perIH = perIH - wl.R + wl.sH;
		perIH = DIVCEIL(perIH, wl.sH * part.H) * wl.sH - wl.sH + wl.R;
	}
	len_t perIW = ifm_range.w.size();
	if (part.W != 1) {
		perIW = perIW - wl.S + wl.sW;
		perIW = DIVCEIL(perIW, wl.sW * part.W) * wl.sW - wl.sW + wl.S;
	}
	return perB * C * perIH * perIW;
}

vol_t ConvLayer::wgt_part(fmap_range& wgt_range, const PartSch& part) const {
	len_t perB = DIVCEIL(wgt_range.b.size(), part.B);
	len_t C = wgt_range.h.size();
	len_t RS = wgt_range.w.size();
	len_t perK = DIVCEIL(wgt_range.c.size(), part.K);
	return perB * perK * C * RS;
}
bool ConvLayer::fmap_channel_rel() const{
	return false;
}

void GroupConvLayer::Workload::init(){
	ConvLayer::Workload::init();
	assert(G >= 1);
	assert(C%G == 0);
	assert(K%G == 0);
	GC = C/G;
	GK = K/G;
	tot_op /= G;
}

vol_t GroupConvLayer::Workload::fil_size() const{
	return R*S*GC*K;
}

GroupConvLayer::GroupConvLayer(const std::string& _name, const Workload& _wl)
	:ConvLayer(_name, _wl), wl(_wl){
	wl.init();
	ConvLayer::wl = wl;
	ConvLayer::wgt_size = wl.fil_size();
	wgt_shape = fmap_shape(wl.K, wl.GC, wl.R*wl.S);
}

const GroupConvLayer::Workload& GroupConvLayer::get_workload() const{
	return wl;
}

void GroupConvLayer::ofm_to_ifm(fmap_range& ofm_range) const{
	len_t group_Klen = wl.GK;
	len_t group_Clen = wl.GC;
	len_t from_id = ofm_range.c.from / group_Klen;
	len_t to_id = DIVCEIL(ofm_range.c.to, group_Klen);
	ConvLayer::ofm_to_ifm(ofm_range);
	ofm_range.c.from = from_id * group_Clen;
	ofm_range.c.to = to_id * group_Clen;
}

void GroupConvLayer::ofm_to_wgt(fmap_range& ofm_range) const{
	// B = B(1)
	// ofm_range.b = {0, 1};
	// C = K, H = C, W = R*S
	// ofm_range.c = ofm_range.c;
	ofm_range.h = {0, wl.GC};
	ofm_range.w = {0, wl.R*wl.S};
}
FetchSch GroupConvLayer::set_fetch(const PartSch& partSch, vol_t size, len_t B, len_t wgt_B) const {
	// Consider fetching.
	FetchSch fetch(1, 1, 1, 1, 1, 1);

	len_t perK = DIVCEIL(ofm_shape.c, partSch.K);
	len_t perB = DIVCEIL(B, partSch.B);
	len_t perH = DIVCEIL(ofm_shape.h, partSch.H);
	len_t perW = DIVCEIL(ofm_shape.w, partSch.W);

	// Can fetch HW iff sH/sW == R/S
	bool canFetchHW;
	len_t ifmH, ifmW;
	// canFetchHW = ((wl.R == wl.sH) && (wl.S == wl.sW));
	canFetchHW = true;
	ifmH = (perH - 1) * wl.sH + wl.R;
	ifmW = (perW - 1) * wl.sW + wl.S;

	len_t ifmC = wl.G - getGCD(wl.G, partSch.K);
	ifmC = (DIVCEIL(ifmC, partSch.K) + 1) * wl.GC;
	vol_t ifmMax = perB * ifmC * ifmH * ifmW;
	vol_t wgtMax = wl.GC * perK * wl.R * wl.S;
	if (wgt_B) wgtMax *= perB;
	assert(ifmMax + wgtMax > size);

	// If wgt_B, also fetch batch.
	if (wgt_B) {
		ifmMax /= perB;
		wgtMax /= perB;
		fetch.B = perB;
		if (ifmMax + wgtMax <= size) {
			return fetch;
		}
	}

	// Consider fetching on K.
	vol_t wgtMin = wgtMax / perK;
	vol_t ifmNoK = (ifmMax / ifmC) * wl.GC;
	if (ifmNoK + wgtMin <= size) {
		len_t maxK = (size - ifmNoK) / wgtMin;
		maxK = MIN(maxK, wl.GK);
		fetch.K = DIVCEIL(perK, maxK);
		return fetch;
	}

	// Considering fetching on BHW.
	vol_t ifmMin = ifmMax;
	if (!wgt_B) {
		ifmMin = ifmMin / perB;
		if (ifmMin + wgtMax <= size) {
			len_t maxB = (size - wgtMax) / ifmMin;
			fetch.B = DIVCEIL(perB, maxB);
			// Opt: under this B, we can fetch K less times.
			len_t maxK = (size - ifmMin * DIVCEIL(perB, fetch.B)) / wgtMin;
			maxK = MIN(maxK, wl.GK);
			fetch.K = DIVCEIL(perK, maxK);
			return fetch;
		}
		fetch.B = perB;
	}
	if (canFetchHW) {
		// For simplicity we only fetch H.
		ifmMin = (ifmMin / ifmH) * wl.R;
		if (ifmMin + wgtMax <= size) {
			len_t maxIH = (size - wgtMax) / (ifmMin / wl.R);
			len_t maxH = (maxIH - wl.R) / wl.sH + 1;
			fetch.H = DIVCEIL(perH, maxH);
			// Opt: under this HW, we can fetch K less times.
			len_t curIH = (DIVCEIL(perH, fetch.H) - 1) * wl.sH + wl.R;
			len_t maxK = (size - (ifmMin / wl.R) * curIH) / wgtMin;
			maxK = MIN(maxK, wl.GK);
			fetch.K = DIVCEIL(perK, maxK);
			return fetch;
		}
	}

	// Fetch both.
	ifmMin = (ifmMin / ifmC) * wl.GC;
	if (ifmMin + wgtMin > size) {
		// can't fetch both.
		fetch.clear();
		return fetch;
	}

	// For simplicity, only consider muli-fetch weight.
	fetch.K = perK;
	fetch.B = 1;
	ifmMin = ifmNoK;

	if (!wgt_B) {
		ifmMin = ifmMin / perB;
		if (ifmMin + wgtMin <= size) {
			len_t maxB = (size - wgtMin) / ifmMin;
			fetch.wgtFetch = fetch.B = DIVCEIL(perB, maxB);
			return fetch;
		}
		fetch.wgtFetch = fetch.B = perB;
	}
	if (canFetchHW) {
		// For simplicity we only fetch H.
		ifmMin = (ifmMin / ifmH) * wl.R;
		if (ifmMin + wgtMin <= size) {
			len_t maxIH = (size - wgtMin) / (ifmMin / wl.R);
			len_t maxH = (maxIH - wl.R) / wl.sH + 1;
			fetch.H = DIVCEIL(perH, maxH);
			fetch.wgtFetch *= fetch.H;
			return fetch;
		}
	}
	// Should not be here.
	assert(false);
	return fetch;

}

vol_t GroupConvLayer::ifm_part(fmap_range& ifm_range, const PartSch& part) const {
	vol_t s = ConvLayer::ifm_part(ifm_range, part);
	if (part.K > 1) {
		s /= ifm_range.c.size();
		s *= wl.GC;
	}
	return s;
}

vol_t GroupConvLayer::wgt_part(fmap_range& wgt_range, const PartSch& part) const {
	vol_t s = ConvLayer::wgt_part(wgt_range, part);
	if (part.K > 1) {
		s /= wgt_range.h.size();
		s *= wl.GC;
	}
	return s;
}
bool GroupConvLayer::fmap_channel_rel() const{
	return true;
}

FCLayer::FCLayer(const std::string& _name, const Workload& wl)
	:ConvLayer (_name, [&]{ConvLayer::Workload cwl;
							cwl.C = wl.C; cwl.K=wl.K; cwl.R=wl.IH; cwl.S=wl.IW; cwl.H=1;
							return cwl;}()){
}

void FCLayer::ofm_to_ifm(fmap_range& ofm_range) const{
	assert(ofm_range.h.from == 0 && ofm_range.w.from == 0 &&
		   ofm_range.h.to == 1 && ofm_range.w.to == 1);
	ofm_range.c = {0, wl.C};
	ofm_range.h.to = wl.R - pad_h;
	ofm_range.w.to = wl.S - pad_w;
}

void LRLayer::Workload::init(){
	W = (W == 0)?H:W;
	S = (S == 0)?R:S;
	sK = (sK == 0)?N:sK;
	sH = (sH == 0)?R:sH;
	sW = (sW == 0)?sH:sW;
	update_op();
	assert(sK<=N && 3 && sW<=S);
	assert(tot_op>0 && sK>0 && sH>0 && sW>0);
}

void LRLayer::Workload::update_op(){
	tot_op = K*H*W*N*R*S;
}

access_t LRLayer::Workload::calc_op(len_t batch_size) const{
	return tot_op*batch_size;
}

LRLayer::LRLayer(const std::string& _name, const Workload& _wl):
	Layer(_name), wl(_wl){
	wl.init();
	ifm_shape = fmap_shape((wl.K - 1) * wl.sK + wl.N, (wl.H - 1) * wl.sH + wl.R, (wl.W - 1) * wl.sW + wl.S);
	ofm_shape = fmap_shape(wl.K, wl.H, wl.W);
	// wgt_shape = fmap_shape(0, 0, 0);
}

const LRLayer::Workload& LRLayer::get_workload() const{
	return wl;
}

const fmap_shape& LRLayer::real_ifmap_shape() const{
	return padded_ifm_shape;
}

vol_t LRLayer::weight_size() const{
	return 0;
}

access_t LRLayer::get_num_op(len_t batch_size) const{
	return wl.calc_op(batch_size);
}

void LRLayer::ofm_to_wgt(fmap_range& ofm_range) const{
	(void) ofm_range;
}
FetchSch LRLayer::set_fetch(const PartSch& partSch, vol_t size, len_t B, len_t wgt_B) const {
	/*
	// No need to fetch.
	(void) partSch;
	(void) size;
	(void) B;
	(void) wgt_B;
	FetchSch f;
	f.clear();
	*/
	// For simplicity, take all K.
	len_t perK = DIVCEIL(ofm_shape.c, partSch.K);
	FetchSch f(perK, 1, 1, 1, 1, 1);
	if (perK == 1) f.clear();
	return f;
}

vol_t LRLayer::ifm_part(fmap_range& ifm_range, const PartSch& part) const {
	(void)ifm_range;
	(void)part;
	assert(part.size() == part.K);
	return DIVCEIL(ifm_range.c.size(), part.K) * ifm_range.b.size() \
		* ifm_range.h.size() * ifm_range.w.size();
}

vol_t LRLayer::wgt_part(fmap_range& wgt_range, const PartSch& part) const {
	(void)wgt_range;
	(void)part;
	assert(false); // Not implemented.
	return 0;
}
bool LRLayer::fmap_channel_rel() const{
	return true;
}

PoolingLayer::PoolingLayer(const std::string& _name, const PoolingLayer::Workload& wl)
	:LRLayer(_name, [&]{LRLayer::Workload lwl;
	lwl.N=1; lwl.K = wl.K; lwl.H = wl.H; lwl.W = wl.W; lwl.R=wl.R;
	lwl.S=wl.S; lwl.sH = wl.sH; lwl.sW = wl.sW;
	return lwl;}()){}

bool PoolingLayer::set_padded_ifm(const fmap_shape& padded_shape){
	if(padded_shape.h > ifm_shape.h
	|| padded_shape.w > ifm_shape.w
	|| padded_shape.c != ifm_shape.c) return false;

	len_t tot_ph = ifm_shape.h - padded_shape.h;
	len_t tot_pw = ifm_shape.w - padded_shape.w;
	if(tot_ph > 2*(wl.R - 1) || tot_pw > 2*(wl.S - 1)){
		return false;
	}
	pad_h = tot_ph/2;
	pad_w = tot_pw/2;
	padded_ifm_shape = padded_shape;
	padded_ifm_shape.update_size();
	return true;
}

void PoolingLayer::ofm_to_ifm(fmap_range& ofm_range) const{
	ofm_range.h.from = ofm_range.h.from * wl.sH;
	ofm_range.h.from = (ofm_range.h.from > pad_h)?(ofm_range.h.from - pad_h):0;
	ofm_range.w.from = ofm_range.w.from * wl.sW;
	ofm_range.w.from = (ofm_range.w.from > pad_w)?(ofm_range.w.from - pad_w):0;
	ofm_range.h.to = (ofm_range.h.to-1) * wl.sH + wl.R - pad_h;
	ofm_range.w.to = (ofm_range.w.to-1) * wl.sW + wl.S - pad_w;
	ofm_range.h.to = MIN(ofm_range.h.to, padded_ifm_shape.h);
	ofm_range.w.to = MIN(ofm_range.w.to, padded_ifm_shape.w);
}

EltwiseLayer::EltwiseLayer(const std::string& _name, const EltwiseLayer::Workload& wl)
	:LRLayer(_name, [&]{LRLayer::Workload lwl;
	lwl.N=wl.N; lwl.K = wl.K; lwl.H = wl.H; lwl.W = wl.W; lwl.R=1;
	return lwl;}()){
	padded_ifm_shape = ifm_shape;
	pad_h = pad_w = 0;
}

bool EltwiseLayer::set_padded_ifm(const fmap_shape& padded_shape){
	return padded_shape == padded_ifm_shape;
}

void EltwiseLayer::ofm_to_ifm(fmap_range& ofm_range) const{
	(void) ofm_range;
}

PTPLayer::PTPLayer(const std::string& _name, const PTPLayer::Workload& wl)
	:LRLayer(_name, [&]{LRLayer::Workload lwl;
	lwl.K = wl.K; lwl.H = wl.H; lwl.W = wl.W; lwl.N=1; lwl.R=1;
	return lwl;}()){
	padded_ifm_shape = ifm_shape;
	pad_h = pad_w = 0;
}

bool PTPLayer::set_padded_ifm(const fmap_shape& padded_shape){
	return padded_shape == padded_ifm_shape;
}

void PTPLayer::ofm_to_ifm(fmap_range& ofm_range) const{
	(void) ofm_range;
}

TransposeLayer::Workload::Workload(){
	for(int i=0; i<dim::NUM; ++i){
		order[i] = static_cast<dim>(i);
	}
}

void TransposeLayer::Workload::init(){
	bool used[dim::NUM];
	memset(used, 0, sizeof(bool)*dim::NUM);
	for(int i=0; i<dim::NUM; ++i){
		assert(!used[order[i]]);
		used[order[i]] = true;
	}
}

fmap_range::dim_range& TransposeLayer::Workload::get_origin_dim(fmap_range& range, dim d) const{
	switch(order[d]){
	case dim::C:
		return range.c;
	case dim::H:
		return range.h;
	case dim::W:
		return range.w;
	default:
		assert(false);
		return range.b;
	}
}

TransposeLayer::TransposeLayer(const std::string& _name, const TransposeLayer::Workload& _wl)
	:LRLayer(_name, [&]{LRLayer::Workload lwl;
	lwl.K = _wl.K; lwl.H = _wl.H; lwl.W = _wl.W; lwl.N=1; lwl.R=1;
	return lwl;}()), wl(_wl){
	fmap_range ifm_range(ofm_shape);
	TransposeLayer::ofm_to_ifm(ifm_range);
	ifm_shape = {ifm_range.c.to, ifm_range.h.to, ifm_range.w.to};
	padded_ifm_shape = ifm_shape;
	pad_h = pad_w = 0;
}

bool TransposeLayer::set_padded_ifm(const fmap_shape& padded_shape){
	return padded_shape == padded_ifm_shape;
}

void TransposeLayer::ofm_to_ifm(fmap_range& ofm_range) const{
	fmap_range _ofm_range = ofm_range;
	wl.get_origin_dim(ofm_range, dim::C) = _ofm_range.c;
	wl.get_origin_dim(ofm_range, dim::H) = _ofm_range.h;
	wl.get_origin_dim(ofm_range, dim::W) = _ofm_range.w;
}

