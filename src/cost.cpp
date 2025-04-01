#include <string>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <functional>
#include <math.h> 
#include "cost.h"
#define _USE_MATH_DEFINES     
 
namespace spec {
    const std::vector<std::string> nodes = {"3", "5", "7", "12", "14", "20", "28", "40", "55"};
    
    const double NRE_scale_factor_module = 0.5;
    const double NRE_scale_factor_chip = 0.3;
    
    std::unordered_map<std::string, double> Cost_NRE;
    std::unordered_map<std::string, double> Module_NRE_Cost_Factor;
    std::unordered_map<std::string, double> Chip_NRE_Cost_Factor;
    std::unordered_map<std::string, double> Chip_NRE_Cost_Fixed;
    std::unordered_map<std::string, double> Cost_Wafer_Die;
    std::unordered_map<std::string, double> Defect_Density_Die;
    
    // the parameters of the wafer
    const double wafer_diameter = 300;
    const double scribe_lane = 0.2;
    const double edge_loss = 5;
    const double critical_level = 10;
    
    // the parameters of the OS(organic substrate)
    const double os_area_scale_factor = 4.0;
    const double os_NRE_cost_factor = 3e3;
    const double os_NRE_cost_fixed = 3e5;
    const double cost_factor_os = 0.005;
    const double bonding_yield_os = 0.99;
    const double c4_bump_cost_factor = 0.005;
    
    // the parameters of the FO(Intigrated Fan-Out)
    const double cost_wafer_rdl = 1200;
    const double defect_density_rdl = 0.05;
    const double rdl_area_scale_factor = 1.2;
    const double critical_level_rdl = 3;
    const double bonding_yield_rdl = 0.98;
    const double fo_NRE_cost_factor = 7.5e6;
    const double fo_NRE_cost_fixed = 7.5e6;
    
    // the parameters of the SI(Silicon Interposer)
    const double defect_density_si = 0.06;
    const double si_area_scale_factor = 1.1;
    const double critical_level_si = 6;
    const double bonding_yield_si = 0.95;
    const double u_bump_cost_factor = 0.01;
    const double cost_wafer_si = Cost_Wafer_Die["55"];
    double si_NRE_cost_factor = 0;  
    double si_NRE_cost_fixed = 0;  
    void initialize() {
        std::map<std::string, double> defect_density = {
            {"3", 0.2}, {"5", 0.11}, {"7", 0.09}, {"12",0.08},
            //{"10", 0.08},
            {"14", 0.08}, {"20", 0.07}, {"28", 0.07}, {"40", 0.07}, {"55", 0.07}
        };
        
        std::map<std::string, double> wafer_cost = {
            {"3", 30000}, {"5", 16988}, {"7", 9346}, {"12", 5992},
            // {"10", 5992},
            {"14", 3984}, {"20", 3677}, {"28", 2891}, {"40", 2274}, {"55", 1937}
        };
        
        std::map<std::string, double> nre_cost = {
            {"3", 100e7}, {"5", 54.2e7}, {"7", 29.8e7}, {"12",17.4e7},
            // {"10", 17.4e7},
            {"14", 10.6e7}, {"20", 7e7}, {"28", 5.1e7}, {"40", 3.8e7}, {"55", 2.8e7}
        };
        for(const auto& node : nodes) {
            Cost_NRE[node] = nre_cost[node];
            Module_NRE_Cost_Factor[node] = NRE_scale_factor_module * Cost_NRE[node] / 300;
            Chip_NRE_Cost_Factor[node] = NRE_scale_factor_chip * Cost_NRE[node] / 300;
            Chip_NRE_Cost_Fixed[node] = (1 - NRE_scale_factor_module - NRE_scale_factor_chip) * Cost_NRE[node];
            Cost_Wafer_Die[node] = wafer_cost[node];
            Defect_Density_Die[node] = defect_density[node];
        }
        
        // setting the parameters of the SI
        si_NRE_cost_factor = Chip_NRE_Cost_Factor["55"] * 1.2;
        si_NRE_cost_fixed = Chip_NRE_Cost_Fixed["55"] * 1.2;
    }
}

ChipConfig::ChipConfig(std::string n, std::string cn, std::string pt, double a, int q)
    : node(n), chip_name(cn), package_type(pt), area(a), num(q) {}

// Class Chip
Chip::Chip(const std::string& name, const std::string& node, double area)
    : name(name), node(node), area(area), knownNRE(0) {
    cost_factor = spec::Chip_NRE_Cost_Factor[node];
    fixed = spec::Chip_NRE_Cost_Fixed[node];
}
size_t Chip::Hash::operator()(const Chip& c) const {
    return std::hash<std::string>()(c.name) ^ std::hash<std::string>()(c.node) ^ std::hash<double>()(c.area);
}
bool Chip::operator==(const Chip& other) const {
    return name == other.name && node == other.node && area == other.area;
}
double Chip::getArea() const { return area; }
std::string Chip::getNode() const { return node; }
void Chip::setFactor(double factor) { cost_factor = factor; }
void Chip::setFixed(double fixed_val) { fixed = fixed_val; }
void Chip::setNRE(double n) { knownNRE = n; }
double Chip::NRE() const {
    return (knownNRE != 0) ? knownNRE : (area * cost_factor + fixed);
}
double Chip::die_yield() const {
    return pow(1 + (spec::Defect_Density_Die[node] / 100 * area / spec::critical_level), -spec::critical_level);
}
double Chip::N_KGD() const {
    return N_die_total() * die_yield();
}
double Chip::N_die_total() const {
    double Area_chip = area + 2 * spec::scribe_lane * sqrt(area) + pow(spec::scribe_lane, 2);
    double N_total = (M_PI * pow(spec::wafer_diameter/2 - spec::edge_loss, 2) / Area_chip) 
                   - (M_PI * (spec::wafer_diameter - 2 * spec::edge_loss) / sqrt(2 * Area_chip));
    return N_total;
}
double Chip::cost_raw_die() const {
    return spec::Cost_Wafer_Die[node] / N_die_total();
}
double Chip::cost_KGD() const {
    return spec::Cost_Wafer_Die[node] / N_KGD();
}
double Chip::cost_defect() const {
    return cost_KGD() - cost_raw_die();
}
std::tuple<double, double> Chip::cost_RE() const {
    return {cost_raw_die(), cost_defect()};
}
// Class Package
Package::Package(const std::string& name, const std::unordered_map<Chip, int, Chip::Hash>& chips)
    : name(name), chips(chips) {}
size_t Package::Hash::operator()(const Package& p) const {
    size_t hash = std::hash<std::string>()(p.name) ^ std::hash<double>()(p.area());
    return hash;
}
bool Package::operator==(const Package& other) const {
    return name == other.name && area() == other.area();
}
int Package::chip_num() const {
    int count = 0;
    for (const auto& pair : chips) {
        count += pair.second;
    }
    return count;
}
double Package::cost_chips() {
    auto [raw, defect, pkg_raw, pkg_defect, wasted] = cost_RE();
    return raw + defect;
}
double Package::cost_package() {
    auto [raw, defect, pkg_raw, pkg_defect, wasted] = cost_RE();
    return pkg_raw + pkg_defect + wasted;
}
double Package::cost_total_system() {
    return cost_chips() + cost_package();
}

// class OS
OS::OS(const std::string& name, const std::unordered_map<Chip, int, Chip::Hash>& chips)
    : Package(name, chips) {}
double OS::interposer_area() const {
    throw std::runtime_error("No interposer in organic substrate package");
}
double OS::area() const {
    double total_area = 0;
    for (const auto& pair : chips) {
        total_area += pair.first.getArea() * pair.second;
    }
    return total_area * spec::os_area_scale_factor;
}
double OS::NRE() {
    double factor;
    if (chip_num() == 1) {
        factor = 1.0;
    } else if (area() > 30*30) {
        factor = 2.0;
    } else if (area() > 17*17) {
        factor = 1.75;
    } else {
        factor = 1.5;
    }
    return area() * spec::os_NRE_cost_factor * factor + spec::os_NRE_cost_fixed;
}
double OS::cost_raw_package() {
    double factor;
    if (chip_num() == 1) {
        factor = 1.0;
    } else if (area() > 30*30) {
        factor = 2.0;
    } else if (area() > 17*17) {
        factor = 1.75;
    } else {
        factor = 1.5;
    }
    return area() * spec::cost_factor_os * factor;
}
std::tuple<double, double, double, double, double> OS::cost_RE() {
    double raw_chips = 0, defect_chips = 0;
    for (const auto& [chip, num] : chips) {
        raw_chips += (chip.cost_raw_die() + chip.getArea() * spec::c4_bump_cost_factor) * num;
        defect_chips += chip.cost_defect() * num;
    }
    
    double pkg_defect = cost_raw_package() * (1/pow(spec::bonding_yield_os, chip_num()) - 1);
    double wasted = (raw_chips + defect_chips) * (1/pow(spec::bonding_yield_os, chip_num()) - 1);
    
    return {raw_chips, defect_chips, cost_raw_package(), pkg_defect, wasted};
}

// class Advanced
Advanced::Advanced(const std::string& name,
                   const std::unordered_map<Chip, int, Chip::Hash>& chips,
                   double nre_factor, double nre_fixed,
                   double wafer_cost, double defect_den, int crit_level,
                   double bond_yield, double area_scale, int chip_last)
    : Package(name, chips),
      NRE_cost_factor(nre_factor), NRE_cost_fixed(nre_fixed),
      wafer_cost(wafer_cost), defect_density(defect_den),
      critical_level(crit_level), bonding_yield(bond_yield),
      area_scale_factor(area_scale), chip_last(chip_last) {}
double Advanced::interposer_area() const {
    double total = 0;
    for (const auto& [chip, num] : chips) {
        total += chip.getArea() * num;
    }
    return total * area_scale_factor;
}
double Advanced::area() const {
    return interposer_area() * spec::os_area_scale_factor;
}
double Advanced::NRE() {
    return interposer_area() * NRE_cost_factor + NRE_cost_fixed + area() * spec::cost_factor_os;
}
double Advanced::package_yield() const {
    return pow(1 + (defect_density/100 * interposer_area() / critical_level), -critical_level);
}
double Advanced::N_package_total() const {
    double area = interposer_area() + 2*spec::scribe_lane*sqrt(interposer_area()) + pow(spec::scribe_lane, 2);
    return (M_PI * pow(spec::wafer_diameter/2 - spec::edge_loss, 2) / area)
         - (M_PI * (spec::wafer_diameter - 2*spec::edge_loss) / sqrt(2*area));
}
double Advanced::cost_interposer() const {
    return wafer_cost / N_package_total() + interposer_area() * spec::c4_bump_cost_factor;
}
double Advanced::cost_substrate() const {
    return area() * spec::cost_factor_os;
}
double Advanced::cost_raw_package() {
    return cost_interposer() + cost_substrate();
}
std::tuple<double, double, double, double, double> Advanced::cost_RE() {
    double raw_chips = 0, defect_chips = 0;
    for (const auto& [chip, num] : chips) {
        raw_chips += (chip.cost_raw_die() + chip.getArea() * spec::u_bump_cost_factor) * num;
        defect_chips += chip.cost_defect() * num;
    }
    double y1 = package_yield();
    double y2 = pow(bonding_yield, chip_num());
    double y3 = spec::bonding_yield_os;
    double pkg_defect, wasted;
    if (chip_last == 1) {
        pkg_defect = cost_interposer() * (1/(y1*y2*y3) - 1) + cost_substrate() * (1/y3 - 1);
        wasted = (raw_chips + defect_chips) * (1/(y2*y3) - 1);
    } else {
        pkg_defect = cost_interposer() * (1/(y1*y3) - 1) + cost_substrate() * (1/y3 - 1);
        wasted = (raw_chips + defect_chips) * (1/(y1*y3) - 1);
    }
    return {raw_chips, defect_chips, cost_raw_package(), pkg_defect, wasted};
}
// class FO
FO::FO(const std::string& name, const std::unordered_map<Chip, int, Chip::Hash>& chips, int chip_last)
    : Advanced(name, chips,
               spec::fo_NRE_cost_factor, spec::fo_NRE_cost_fixed,
               spec::cost_wafer_rdl, spec::defect_density_rdl,
               spec::critical_level_rdl, spec::bonding_yield_rdl,
               spec::rdl_area_scale_factor, chip_last=1) {}

// class SI
SI::SI(const std::string& name, const std::unordered_map<Chip, int, Chip::Hash>& chips)
    : Advanced(name, chips,
               spec::si_NRE_cost_factor, spec::si_NRE_cost_fixed,
               spec::cost_wafer_si, spec::defect_density_si,
               spec::critical_level_si, spec::bonding_yield_si,
               spec::si_area_scale_factor, 1) {}

// std::unique_ptr<Package> SoC(const std::string& name, const std::string& node, 
//                             const std::unordered_map<std::string, double>& modules, 
//                             const std::string& package_type = "OS") {
//     // Assuming modules calculation for chip area
//     double area = 0;
//     for (const auto& [mod, val] : modules) area += val;
//     Chip chip(name, node, area);

//     std::unordered_map<Chip, int, Chip::Hash> chip_map{{chip, 1}};

//     if (package_type == "OS") {
//         return std::make_unique<OS>(name, chip_map);
//     } else if (package_type == "FO") {
//         return std::make_unique<FO>(name, chip_map);
//     } else if (package_type == "SI") {
//         return std::make_unique<SI>(name, chip_map);
//     }
//     throw std::invalid_argument("Invalid package type");
// }

// Function to calculate the cost of the system
std::tuple<double, double, double> calculate_cost(const std::vector<ChipConfig>& configs) {
    spec::initialize();
    double total_cost = 0;
    double total_cost_chip = 0;
    double total_cost_package = 0; 

    std::unordered_map<std::string, std::vector<ChipConfig>> package_groups;

    for (const auto& cfg : configs) {
        package_groups[cfg.package_type].push_back(cfg);
    }

    for (const auto& [pkg_type, config_list] : package_groups) {
        std::unordered_map<Chip, int, Chip::Hash> chips_in_group;

        for (const auto& cfg : config_list) {
            Chip chip(cfg.chip_name, cfg.node, cfg.area);
            chips_in_group[chip] += cfg.num;
        }

        std::unique_ptr<Package> pkg;
    
    // // 按封装类型分组
    // std::unordered_map<std::string, std::unordered_map<Chip, int, Chip::Hash>> package_groups;
    
    // for (const auto& cfg : configs) {
    //     Chip chip(cfg.chip_name, cfg.node, cfg.area);
    //     package_groups[cfg.package_type][chip] += cfg.num;
    // }
    
    // // 计算每个封装组的成本
    // for (const auto& [pkg_type, chips] : package_groups) {
    //     std::unique_ptr<Package> pkg;
        
        if (pkg_type == "OS") {
            pkg = std::make_unique<OS>("OS_Package", chips_in_group);
        } else if (pkg_type == "FO") {
            pkg = std::make_unique<FO>("FO_Package", chips_in_group);
        } else if (pkg_type == "SI") {
            pkg = std::make_unique<SI>("SI_Package", chips_in_group);
        } else {
            throw std::invalid_argument("Unknown package type: " + pkg_type);
        }
        
        auto [raw, defect, pkg_raw, pkg_defect, wasted] = pkg->cost_RE();
        //total_cost += (raw + defect + pkg_raw + pkg_defect + wasted);
        double cost_chip = 0;
        double cost_package = 0;

        cost_chip = raw + defect;
        cost_package = pkg_raw + pkg_defect + wasted;
        total_cost += (cost_chip + cost_package);
        total_cost_chip += cost_chip;  // cost_chips
        total_cost_package += cost_package;  // cost_package
    }
    return {total_cost, total_cost_chip, total_cost_package};
}

// int main() {
//     std::vector<ChipConfig> configs = {
//         {"7", "CPU", "FO", 100, 1},   // 28nm CPU芯片使用FO封装
//         {"7", "IO", "FO", 50, 3},
//         {"12", "IO", "OS", 50, 2},     // 40nm IO芯片使用OS封装 
//         //{"55", "HBM", "SI", 200, 4}    // 55nm HBM使用SI封装
//     };
//     auto [total_cost, total_cost_chip, total_cost_package] = calculate_cost(configs);
//     std::cout << "Total System Cost: $" << total_cost_chip << std::endl;
//     std::cout << "Total System Cost: $" << total_cost_package << std::endl;
//     std::cout << "Total System Cost: $" << total_cost << std::endl;
    
//     return 0;
// }