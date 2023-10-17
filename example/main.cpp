#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <boost/program_options.hpp>

#include "cholesky_qr.hpp"


void conflicting_options(const boost::program_options::variables_map& vm, 
                         const char* opt1, const char* opt2)
{
    if (vm.count(opt1) && !vm[opt1].defaulted() 
        && vm.count(opt2) && !vm[opt2].defaulted())
        throw std::logic_error(std::string("Conflicting options '") 
                          + opt1 + "' and '" + opt2 + "'.\n"\
                          +"Use either block size 'b' or number of blocks 'bn'");
}


int main(int argc, char** argv) {

    //----- cli program options -----//
    using namespace boost::program_options; 
    options_description desc_commandline("Program options"); 

    desc_commandline.add_options()
        ("m", value<std::int64_t>()->default_value(10000), "Number of rows (default: 10000")
        ("n", value<std::int64_t>()->default_value(1000), "Number of columns (default: 1000")
        ("b", value<std::int64_t>()->default_value(50), "Panel size (default: 50)")
        ("bn", value<std::int64_t>(), "Number of panels")
        ("k", value<int>(), "Exponent of scientific notation of cond number of matrix")
        ("input", value<std::string>(), "Name of input bin matrix")
        ("validate,v", bool_switch()->default_value(false), "validation of the implementation (default: false)")
        ("help,h","Print this message");

 
    variables_map vm;
    store(parse_command_line(argc, argv, desc_commandline), vm);
    notify(vm);

    if (vm.count("help")) {
        std::cout << desc_commandline << std::endl;
        return 1;
    }

    //conflicting_options(vm, "b", "bn");


    // Matrix size
    std::int64_t m = vm["m"].as<std::int64_t>();
    std::int64_t n = vm["n"].as<std::int64_t>();
    std::int64_t block_size = vm["b"].as<std::int64_t>();

    std::int64_t panel_size;
    if(!vm.count("b") && !vm.count("bn")) {
        std::cout << "Set either panel size 'b' or number of panels 'bn'\n";
        exit(1);
    }
    else if(vm.count("b")) {
        panel_size = vm["b"].as<std::int64_t>();
    }
    else {
        panel_size = ceil((double)n / vm["bn"].as<std::int64_t>());
    }

    const char* input_matrix_name_str = vm["input"].as<std::string>().c_str();
    //bool validate = vm["validate"].as<bool>();
    //----- cli program options -----//

#ifdef LOOKAHEAD
    //same api for cpu and gpu versions
    cqr::qr2bgsloohahead algorithm(m, n, block_size);

#elif GSCHOL
    cqr::gschol algorithm(m, n, block_size);
#else 
    cqr::qr2bgs algorithm(m, n, block_size);
#endif
    algorithm.InputMatrix(input_matrix_name_str);
    algorithm.Start();
    return 0;
}
