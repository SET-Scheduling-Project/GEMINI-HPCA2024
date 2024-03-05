#ifndef CORE_H
#define CORE_H

#include <cstdint>
#include "util.h"

class Core{
public:
	typedef std::uint16_t numMac_t;
	const numMac_t mac_num, LR_mac_num;
	const energy_t LR_mac_cost;

	struct Buffer{
		vol_t Size;
		energy_t RCost, WCost;
		bw_t RBW, WBW;
		Buffer(vol_t _size = 0, energy_t _rCost = 0, bw_t _rBW = 256, energy_t _wCost = 0, bw_t _wBW = 0);
	};
	Core(numMac_t _mac_num, numMac_t _LR_mac_num, energy_t _LR_mac_cost);
	virtual const Buffer& ubuf() const = 0;
};

class PolarCore : public Core{
public:
	typedef std::uint8_t vmac_t;
	struct PESetting{
		vmac_t vecSize, laneNum;
		energy_t MACCost;
		PESetting(vmac_t _vecSize, vmac_t _laneNum, energy_t _macCost);
	};
	typedef std::uint8_t numpe_t;
	struct Bus{
		numpe_t aLen,oLen,totNum; // x for act. y for wgt.
		energy_t hopCost;
		bw_t busBW;
		Bus(numpe_t _aLen, numpe_t _oLen, energy_t _hopCost,bw_t _busBW);
	};
	struct Buffers{
		// Notice: there are 8 wl1 in each Polar Core.
		const Buffer al1, wl1, ol1, al2, wl2, ol2, ul3;
	};
	const PESetting pes;
	const Bus bus;
	const Buffer al1, wl1, ol1, al2, wl2, ol2, ul3;
public:
	PolarCore(const PESetting& _pes, const Bus& _noc,
			  const Buffer& _al1,
			  const Buffer& _wl1, const Buffer& _ol1,
			  const Buffer& _al2, const Buffer& _wl2,
			  const Buffer& _ol2, const Buffer& _ul3,
			  numMac_t _LR_mac_num, energy_t _LR_mac_cost);
	virtual const Core::Buffer& ubuf() const;
};
class EyerissCore : public Core {
public:
	typedef std::uint8_t vmac_t;
	struct PESetting {
		vmac_t Xarray, Yarray;
		energy_t MacCost;
		PESetting(vmac_t _Xarray, vmac_t _Yarray, energy_t _MacCost);
	};
	struct Bus {
		energy_t BusCost;//global bus
		bw_t BusBW;
		Bus(energy_t _BusCost, bw_t _BusBW);
	};
	const PESetting pes;
	const Bus ibus, wbus, pbus;
	const Buffer al1, wl1, pl1, ul2;
public:
	EyerissCore(const PESetting& _pes, const Bus& _ibus, const Bus& _wbus, const Bus& _pbus,
		const Buffer& _al1, const Buffer& _wl1,
		const Buffer& _ol1, const Buffer& _ul2,
		numMac_t _LR_mac_num, energy_t _LR_mac_cost
	);
	virtual const Core::Buffer& ubuf() const;
};
#endif // CORE_H
