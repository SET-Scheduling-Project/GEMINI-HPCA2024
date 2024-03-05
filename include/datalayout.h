#ifndef DATALAYOUT_H
#define DATALAYOUT_H

#include <memory>
#include <vector>
#include "util.h"

class BufferUsage;
class Layer;
class StdLayerEngine;
class FastLayerEngine;
struct FetchSch;
struct PartSch;
struct PlaceSch;
//#include "layer.h"
//#include "bufferusage.h"
//#include "layerengine.h"
//#include "partition.h"
//#include "placement.h"


typedef std::vector<fmap_range> fmap_list;

/*
 *  @brief A class that encodes a set of fmap_range that
 *  is cut from a whole fmap_range by a PartSch.
 *
 *  We only stores the parted dim_range for each dim, then
 *  group it into a UniqueDataLayout.

class LogicalDataLayout{
private:
	typedef fmap_range::dim_range dim_r;
	dim_r* r_list[4];
	cidx_t r_len[4];
	cidx_t r_step[4];
};
*/

/**
 * @brief Base class for all data layouts
 */
class DataLayout{
	friend StdLayerEngine;
	friend FastLayerEngine;
public:
	typedef cidx_t dataLen_t;
	/**
	 * @brief The Entry class
	 * Describes that "1/divN" of "range" is stored/needed by
	 * the array "tiles" of length "numTile".
	 */
	struct Entry{
		const fmap_range& range;
		const pos_t* tiles;
		dataLen_t numTile;
		dataLen_t divN;
	};
	struct UniqueEntry{
		const fmap_range& range;
		const pos_t& tile;
		dataLen_t divN;
	};
protected:
	vol_t totVolume, maxVolume;
	len_t multFactor;
	fmap_range MaxRange;
	void update(fmap_range& range);
public:
	DataLayout();
	vol_t totalSize() const;
	vol_t maxRange() const;
	fmap_range MaxRange_() const;
	void sizeMult(len_t num);
	void clear();
	bool update(BufferUsage& usage, const Layer& l, const FetchSch& fetch, bool isIfm) const;
	virtual DataLayout* clone() const = 0;
	virtual void finalize() = 0;
	virtual void reset() = 0;
	virtual dataLen_t totLength() const = 0;
	virtual dataLen_t rangeLength() const = 0;
	dataLen_t bcastLength() const;
	virtual Entry at(dataLen_t idx) const = 0;
	virtual UniqueEntry operator[](dataLen_t idx) const = 0;
	virtual ~DataLayout() = default;
};

class UniqueLayout : public DataLayout{
protected:
	dataLen_t len;
public:
	class Iterator{
		dataLen_t i;
		const UniqueLayout& layout;
	public:
		Iterator(const UniqueLayout& _layout, dataLen_t _i = 0);
		Iterator& operator++();
		std::pair<fmap_range, pos_t> operator*() const;
		bool operator!=(const Iterator& other) const;
	};
	UniqueLayout(dataLen_t _len);
	// virtual void init(const PartSch& sch) = 0;
	Iterator begin() const;
	Iterator end() const;
	virtual UniqueLayout* clone() const override = 0;
	virtual void finalize() override = 0;
	virtual void reset() override = 0;
	virtual dataLen_t totLength() const override final;
	virtual dataLen_t rangeLength() const override final;
	virtual Entry at(dataLen_t idx) const override final;
	virtual UniqueEntry operator[](dataLen_t idx) const override = 0;
};

class StdDataLayout : public DataLayout{
	friend StdLayerEngine;
	friend FastLayerEngine;
private:
	dataLen_t range_len, bcast_len, tot_len, bcast_step, bcast_down;
	std::unique_ptr<fmap_range[]> rangeArr;
	std::unique_ptr<pos_t[]> contPosArr;
	pos_t* posArr;
public:
	StdDataLayout(dataLen_t _len, pos_t* _posArr);
	// void init(const PartSch& sch, UniqueDataLayout& ofm_layout);
	//virtual const_iterator begin() const;
	//virtual const_iterator end() const;
	virtual DataLayout* clone() const override;
	virtual void finalize() override;
	virtual void reset() override;
	virtual dataLen_t totLength() const override;
	virtual dataLen_t rangeLength() const override;
	virtual Entry at(dataLen_t idx) const override;
	virtual UniqueEntry operator[](dataLen_t idx) const override;
	void setCPosArr();
	void setBcast(dataLen_t _bcastLen, dataLen_t _bcastStep);
	//~StdDataLayout();
};

class StdULayout : public UniqueLayout{
	friend StdLayerEngine;
	friend FastLayerEngine;
private:
	dataLen_t dimLen[4];
	dataLen_t dimStep[4];
	std::unique_ptr<fmap_range[]> rangeArr;
	std::unique_ptr<pos_t[]> localPosArr;
	pos_t* posArr;
public:
	class IntersectIter{
		friend StdULayout;
		dataLen_t from[4], to[4], cur[4];
		const StdULayout& layout;
		// IntersectIter(const UniqueLayout& _layout);
		IntersectIter(dataLen_t _from[], dataLen_t _to[], const StdULayout& _layout);
	public:
		UniqueEntry operator*() const;
		bool isValid() const;
		void next();
	};
	StdULayout(dataLen_t _len, pos_t* _posArr);
	// virtual void init(const PartSch& sch) override;
	virtual UniqueLayout* clone() const override;
	virtual void finalize() override;
	virtual void reset() override;
	void setDims(dataLen_t C, dataLen_t B, dataLen_t H, dataLen_t W);
	virtual UniqueEntry operator[](dataLen_t idx) const override;
	IntersectIter get_intersect(const fmap_range& range, bool noBatch) const;
	std::vector<std::pair<fmap_range, pos_t> > get_intersect_bruteforce(const fmap_range& range, len_t batch_size, len_t range_batch_size) const;
	//~StdUDataLayout();
};
/*
class MemULayout : public UniqueLayout{
	friend StdLayerEngine;
private:
	fmap_range range;
	pos_t* posArr;
public:
	MemULayout(const fmap_range& _range, pos_t* _posArr, len_t memLen);
	// virtual void init(const PartSch& sch) override;
	virtual UniqueLayout* clone() const override;
	virtual void reset() override;
	virtual UniqueEntry operator[](dataLen_t idx) const override;
	//~MemULayout();
};
*/
//class DataLayout: std::vector<std::pair<fmap_range, pos_t>>{
/*
private:
	template<typename ptr>
	class DLIterator{
		using layout_ptr = std::conditional_t<std::is_const_v<ptr>, const DataLayout*, DataLayout*>;
		const layout_ptr layout;
		cidx_t it;
		struct X{
			int x;
			int y;
		};
	public:
		struct ChipInfo{
			using range_ref = std::conditional_t<std::is_const_v<ptr>, const fmap_range&, fmap_range&>;
			using pos_ref = std::conditional_t<std::is_const_v<ptr>, const pos_t&, pos_t&>;
			range_ref range;
			pos_ref pos;
		};
		DLIterator(layout_ptr _layout);
		bool operator==(const DLIterator<ptr>& other) const;
		bool operator!=(const DLIterator<ptr>& other) const;
		DLIterator<ptr>& operator++();
		ChipInfo operator*() const;
	};
public:
	typedef DLIterator<DataLayout> iterator;
	typedef DLIterator<const DataLayout> const_iterator;
*/
/*
private:
	fmap_range* range;
	pos_t* pos;
	cidx_t range_len, bcast_len, tot_len;
public:
	DataLayout();
	DataLayout(cidx_t len, cidx_t bcast);
	//virtual const_iterator begin() const;
	//virtual const_iterator end() const;
	~DataLayout();
};

class UniqueDataLayout{
private:
	typedef fmap_range::dim_range dim_r;
	dim_r* r_list[4];
	cidx_t r_len[4];
	cidx_t r_step[4];
public:
	UniqueDataLayout();
	UniqueDataLayout(const PartSch& part);
};
*/
#endif // DATALAYOUT_H
