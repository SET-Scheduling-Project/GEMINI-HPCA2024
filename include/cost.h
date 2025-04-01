#ifndef COST_H
#define COST_H

#include <string>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <functional>

#define _USE_MATH_DEFINES  
#include <math.h>        

class Chip;
class Package;
class Advanced;
class OS;
class FO;
class SI;

struct ChipConfig {
    std::string node;
    std::string chip_name;
    std::string package_type;
    double area;
    int num;

    ChipConfig(std::string n, std::string cn, std::string pt, double a, int q);
};

std::tuple<double, double, double> calculate_cost(const std::vector<ChipConfig>& configs);

namespace spec {
    extern const std::vector<std::string> nodes;

    extern const double NRE_scale_factor_module;
    extern const double NRE_scale_factor_chip;

    extern std::unordered_map<std::string, double> Cost_NRE;
    extern std::unordered_map<std::string, double> Module_NRE_Cost_Factor;
    extern std::unordered_map<std::string, double> Chip_NRE_Cost_Factor;
    extern std::unordered_map<std::string, double> Chip_NRE_Cost_Fixed;
    extern std::unordered_map<std::string, double> Cost_Wafer_Die;
    extern std::unordered_map<std::string, double> Defect_Density_Die;
    
    extern const double wafer_diameter;
    extern const double scribe_lane;
    extern const double edge_loss;
    extern const double critical_level;
    
    extern const double os_area_scale_factor;
    extern const double os_NRE_cost_factor;
    extern const double os_NRE_cost_fixed;
    extern const double cost_factor_os;
    extern const double bonding_yield_os;
    extern const double c4_bump_cost_factor;
    
    extern const double cost_wafer_rdl;
    extern const double defect_density_rdl;
    extern const double rdl_area_scale_factor;
    extern const double critical_level_rdl;
    extern const double bonding_yield_rdl;
    extern const double fo_NRE_cost_factor;
    extern const double fo_NRE_cost_fixed;
    
    extern const double defect_density_si;
    extern const double si_area_scale_factor;
    extern const double critical_level_si;
    extern const double bonding_yield_si;
    extern const double u_bump_cost_factor;
    extern const double cost_wafer_si;
    extern double si_NRE_cost_factor;
    extern double si_NRE_cost_fixed;

    void initialize();
}

// Chip类声明
class Chip {
private:
    std::string name;
    std::string node;
    double area;
    double cost_factor;
    double fixed;
    double knownNRE;

public:
    Chip(const std::string& name, const std::string& node, double area);

    struct Hash {
        size_t operator()(const Chip& c) const;
    };

    bool operator==(const Chip& other) const;

    double getArea() const;
    std::string getNode() const;

    void setFactor(double factor);
    void setFixed(double fixed_val);
    void setNRE(double n);

    double NRE() const;
    double die_yield() const;
    double N_KGD() const;
    double N_die_total() const;
    double cost_raw_die() const;
    double cost_KGD() const;
    double cost_defect() const;
    std::tuple<double, double> cost_RE() const;
};

// Package类及其派生类声明
class Package {
protected:
    std::string name;
    std::unordered_map<Chip, int, Chip::Hash> chips;

public:
    Package(const std::string& name, const std::unordered_map<Chip, int, Chip::Hash>& chips);
    virtual ~Package() = default;

    struct Hash {
        size_t operator()(const Package& p) const;
    };

    bool operator==(const Package& other) const;

    int chip_num() const;

    virtual double interposer_area() const = 0;
    virtual double area() const = 0;
    virtual double NRE() = 0;
    virtual double cost_raw_package() = 0;
    virtual std::tuple<double, double, double, double, double> cost_RE() = 0;

    double cost_chips();
    double cost_package();
    double cost_total_system();
};

class OS : public Package {
public:
    OS(const std::string& name, const std::unordered_map<Chip, int, Chip::Hash>& chips);

    double interposer_area() const override;
    double area() const override;
    double NRE() override;
    double cost_raw_package() override;
    std::tuple<double, double, double, double, double> cost_RE() override;
};

class Advanced : public Package {
protected:
    double NRE_cost_factor;
    double NRE_cost_fixed;
    double wafer_cost;
    double defect_density;
    int critical_level;
    double bonding_yield;
    double area_scale_factor;
    int chip_last;

public:
    Advanced(const std::string& name,
             const std::unordered_map<Chip, int, Chip::Hash>& chips,
             double nre_factor, double nre_fixed,
             double wafer_cost, double defect_den, int crit_level,
             double bond_yield, double area_scale, int chip_last);

    double interposer_area() const override;
    double area() const override;
    double NRE() override;
    double package_yield() const;
    double N_package_total() const;
    double cost_interposer() const;
    double cost_substrate() const;
    double cost_raw_package() override;
    std::tuple<double, double, double, double, double> cost_RE() override;
};

class FO : public Advanced {
public:
    FO(const std::string& name, const std::unordered_map<Chip, int, Chip::Hash>& chips, int chip_last = 1);
};

class SI : public Advanced {
public:
    SI(const std::string& name, const std::unordered_map<Chip, int, Chip::Hash>& chips);
};

#endif // COST_H
