#include "core.h"

#include <cassert>

Core::Core(numMac_t _mac_num, numMac_t _LR_mac_num, energy_t _LR_mac_cost):
	mac_num(_mac_num), LR_mac_num(_LR_mac_num), LR_mac_cost(_LR_mac_cost){}

PolarCore::PolarCore(const PESetting& _pes, const Bus& _noc,
					 const Buffer& _al1,
					 const Buffer& _wl1, const Buffer& _ol1,
					 const Buffer& _al2, const Buffer& _wl2,
					 const Buffer& _ol2, const Buffer& _ul3,
					 numMac_t _LR_mac_num, energy_t _LR_mac_cost):
	Core(_noc.aLen * _noc.oLen * _pes.laneNum * _pes.vecSize, _LR_mac_num, _LR_mac_cost),
	pes(_pes),bus(_noc),al1(_al1),wl1(_wl1),ol1(_ol1),
	al2(_al2),wl2(_wl2),ol2(_ol2),ul3(_ul3){
	assert(al2.Size == 0 || al2.Size >= al1.Size * bus.totNum);
	assert(wl2.Size == 0 || wl2.Size >= wl1.Size * bus.totNum * pes.laneNum);
}

const Core::Buffer& PolarCore::ubuf() const{
	return ul3;
}

PolarCore::PESetting::PESetting(vmac_t _vecSize, vmac_t _laneNum, energy_t _macCost):
	vecSize(_vecSize),laneNum(_laneNum),MACCost(_macCost){}

Core::Buffer::Buffer(vol_t _size, energy_t _rCost, bw_t _rBW, energy_t _wCost, bw_t _wBW):
	Size(_size), RCost(_rCost), WCost(_wCost <= 0?_rCost:_wCost),
	RBW(_rBW), WBW(_wBW<=0?_rBW:_wBW){}

PolarCore::Bus::Bus(numpe_t _aLen, numpe_t _oLen, energy_t _hopCost, bw_t _busBW):
	aLen(_aLen),oLen(_oLen),totNum(_aLen*_oLen),
	hopCost(_hopCost),busBW(_busBW){}

EyerissCore::EyerissCore(const PESetting& _pes, const Bus& _ibus, const Bus& _wbus, const Bus& _pbus,
	const Buffer& _al1, const Buffer& _wl1,
	const Buffer& _pl1, const Buffer& _ul2, numMac_t _LR_mac_num, energy_t _LR_mac_cost):
	Core(_pes.Xarray * _pes.Yarray, _LR_mac_num, _LR_mac_cost),
	pes(_pes), ibus(_ibus), wbus(_wbus), pbus(_pbus), al1(_al1), wl1(_wl1), pl1(_pl1),
	ul2(_ul2) {}

const Core::Buffer& EyerissCore::ubuf() const{
	return ul2;
}

EyerissCore::PESetting::PESetting(vmac_t _Xarray, vmac_t _Yarray, energy_t _MacCost) :
	Xarray(_Xarray), Yarray(_Yarray), MacCost(_MacCost){}

EyerissCore::Bus::Bus(energy_t _BusCost, bw_t _BusBW) :
	BusCost(_BusCost), BusBW(_BusBW) {}
