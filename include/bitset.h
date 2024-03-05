#ifndef BITSET_H
#define BITSET_H

#include <bitset>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <vector>

#define FOR_BITSET(var, set) for(Bitset::bitlen_t var = set.first(); var != set.max_size(); var = set.next(var))

#define MAX_BITS_IN_BS 640

class Bitset: private std::bitset<MAX_BITS_IN_BS>{
public:
	typedef std::uint16_t bitlen_t;
private:
	typedef std::bitset<MAX_BITS_IN_BS> std_bs;
	// std::int64_t bits[4];
	Bitset(std_bs&& base);
public:
	Bitset()=default;
	explicit Bitset(bitlen_t bit);
	Bitset(std::initializer_list<bitlen_t> bits);
	Bitset(std::vector<bitlen_t> list);
	bitlen_t count() const;
	// Undefined behaviour if bitset is empty.
	bitlen_t first() const;
	// Returns 0 if there is no next.
	bitlen_t next(bitlen_t bit) const;
	bitlen_t find_nth_1(bitlen_t bit)const;
	bool contains(bitlen_t bit) const;
	void set();
	void set(bitlen_t bit);
	void reset(bitlen_t bit);
	void flip(bitlen_t bit);
	void clear();
	bitlen_t max_size() const;
	Bitset& operator|=(const Bitset& other);
	bool operator==(const Bitset& other) const;
	bool operator[](const bitlen_t bit)const;
	//Bitset& operator|=(bitlen_t other);
	friend std::ostream& operator<<(std::ostream& os, const Bitset& set);
	friend Bitset operator|(const Bitset& lhs, const Bitset& rhs);
};

#undef MAX_BITS_IN_BS

#endif // BITSET_H
