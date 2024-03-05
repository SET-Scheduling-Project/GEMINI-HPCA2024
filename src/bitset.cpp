#include "bitset.h"

/*
Bitset::Bitset(){

}

Bitset::Bitset(Bitset::bitlen_t bit){

}

Bitset& Bitset::operator|=(const Bitset& other){

}

Bitset& Bitset::operator|=(Bitset::bitlen_t other){

}
*/

Bitset::Bitset(std_bs&& base):std_bs(base){}

Bitset::Bitset(Bitset::bitlen_t bit):std_bs(){
	set(bit);
}

Bitset::Bitset(std::initializer_list<bitlen_t> bits):std_bs(){
	for(auto i = bits.begin(); i!= bits.end(); ++i){
		set(*i);
	}
}

Bitset::Bitset(std::vector<bitlen_t> list){
	for(auto i: list){
		set(i);
	}
}

Bitset::bitlen_t Bitset::count() const{
	return static_cast<Bitset::bitlen_t>(std_bs::count());
}

Bitset::bitlen_t Bitset::first() const{
	return static_cast<Bitset::bitlen_t>(_Find_first());
}

Bitset::bitlen_t Bitset::next(Bitset::bitlen_t bit) const{
	return static_cast<Bitset::bitlen_t>(_Find_next(bit));
}
Bitset::bitlen_t Bitset::find_nth_1(Bitset::bitlen_t bit) const {
	bitlen_t i = first();
	for (bitlen_t p=0;p<bit-1;p++){
			i=next(i);
	}
	return i;
}

void Bitset::flip(Bitset::bitlen_t bit){
	std_bs::flip(bit);
}

bool Bitset::contains(Bitset::bitlen_t bit) const{
	return test(bit);
}

void Bitset::set(){
	std_bs::set();
}

void Bitset::set(Bitset::bitlen_t bit){
	std_bs::set(bit);
}

void Bitset::reset(Bitset::bitlen_t bit){
	std_bs::reset(bit);
}

void Bitset::clear(){
	std_bs::reset();
}

Bitset::bitlen_t Bitset::max_size() const{
	return static_cast<Bitset::bitlen_t>(std_bs::size());
}

Bitset& Bitset::operator|=(const Bitset& other){
	std_bs::operator|=(other);
	return *this;
}

bool Bitset::operator==(const Bitset& other) const{
	return std_bs::operator==(other);
}
bool Bitset::operator[](const bitlen_t bit) const {
	return std_bs::operator[](bit);
}
/*
Bitset& Bitset::operator|=(Bitset::bitlen_t other){
	set(other);
	return *this;
}
*/

std::ostream& operator<<(std::ostream& os, const Bitset& set){
	if(set.count() == 0) return os << "()";
	Bitset::bitlen_t bit = set.first();
	os << '(' << bit;
	if(set.count() == 1) return os << ",)";
	while((bit = set.next(bit)) != set.max_size()){
		os << ',' << bit;
	}
	return os << ')';
}

Bitset operator|(const Bitset& lhs, const Bitset& rhs){
	return static_cast<const Bitset::std_bs&>(lhs) | static_cast<const Bitset::std_bs&>(rhs);
}
