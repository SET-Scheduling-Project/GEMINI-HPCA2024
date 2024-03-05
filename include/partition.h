#ifndef PARTITION_H
#define PARTITION_H

#include <vector>
#include "util.h"

class Node;
//#include "network.h"

struct PartSch{
	// TODO: change to cidx_t.
	typedef len_t partlen_t;
	len_t K, B, H, W;
	explicit PartSch()=default;
	PartSch(len_t _K, len_t _B, len_t _H, len_t _W);
	len_t& operator[](std::uint8_t i);
	const len_t& operator[](std::uint8_t i) const;
	vol_t size() const;
	friend std::ostream& operator<<(std::ostream& os, const PartSch& sch);
};

struct FetchSch : public PartSch {
	len_t ifmFetch;
	len_t wgtFetch;
	explicit FetchSch() = default;
	FetchSch(len_t _K, len_t _B, len_t _H, len_t _W, len_t _ifmF, len_t _wgtF);
	operator bool() const;
	void clear();
};

class PartIter;

class PartEngine{
public:
	typedef std::uint16_t factor_t;
private:
	friend PartIter;
	
	typedef std::vector<PartSch> fvec;
	struct num_pair{
		factor_t x,y;
	};

	static fvec factors[MAX_CHIPS+1];
	static double utils[MAX_CHIPS+1][MAX_CHIPS+1];
	double min_util;

	static std::vector<num_pair> factor_num(factor_t n);
	static void init_all();
public:
	PartEngine(double _min_util=0.75);

	PartIter init(cidx_t cluster_size, len_t batch_num, const Node& layer, PartSch& sch, len_t min_cuts);
}extern partEngine;

class PartIter{
	friend PartEngine;
	typedef PartEngine::fvec fvec;
	typedef fvec::const_iterator citer;

	PartSch& curSch;
	citer nextPos, endPos;
	len_t maxB, maxK, maxH, maxW, min_ncut;
	double min_util;
	bool finished;

	PartIter(PartSch& partSch, double _min_util=0.75);

	bool calcUtil(const PartSch& nextSch) const;
	static double calc_util(len_t real, len_t part);
	bool getBestPart(cost_t cost = cost_inf);
	bool calc_util_best(const PartSch& cur_sch);
public:
	bool nextPart(cost_t cost = cost_inf);
	/**
	 * @brief operator bool: same as "eof" in an IO stream.
	 */
	operator bool() const;
};

#endif // PARTITION_H
