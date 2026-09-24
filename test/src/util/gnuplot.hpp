#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <utility>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

#include "util.hpp"
#include "run_cmd.hpp"

namespace {
    namespace gnuplot{
        const std::string output_dir = "data/analyze_gnuplot";

        const std::vector<double> pass_rate_vec = {.8, .9, .95, 1};
        const std::vector<int> color_vec = {1, 6, 4, 7};

        // step counts
        template <class T>
        constexpr int step_cnt = 256;
        template<>
        constexpr int step_cnt<std::uint16_t> = 256 * 256/30;
        template<>
        constexpr int step_cnt<float> = 256;

        // helper functions
        int mkdir(){
            return RunCmd("mkdir -p " + gnuplot::output_dir);
        }

        template<class T>
        long double plot_max(){
            // avoid overflow
            static_assert(
                !std::numeric_limits<T>::is_integer
              || static_cast<unsigned long long int>(std::numeric_limits<T>::max()) * step_cnt<T> < std::numeric_limits<long double>::max()
              , "Could not get max value to plot for the type T"
            );
            return
                std::numeric_limits<T>::is_integer
                    ? std::numeric_limits<T>::max()
                    : 255;
        }

        template<class T>
        void prepare_files(std::string outputName){
            std::string gpInput = outputName + ".input";
            std::string svgOutput = outputName + ".svg";
            std::string gpInputPath = Util::pathCat(gnuplot::output_dir, gpInput);
            std::string title = outputName;
            for(char& c : title) if(c == '_') c = ' ';
            std::ofstream gpInputOfs(gpInputPath);
            gpInputOfs
                << "set terminal svg size 600,480 font \"Arial,10\"" << std::endl
            //    << "set xtics 5" << std::endl
            //    << "set ytics 0.05" << std::endl
                << "set xlabel \"threshold\"" << std::endl
                << "set ylabel \"error rate\"" << std::endl
                << "set grid lw 2" << std::endl
                << "set key box" << std::endl
                << "set title \"" << title << "\"" << std::endl
                << "plot [0:" << plot_max<T>() << "][0:1]";
            for(int i=0; i<pass_rate_vec.size(); ++i){
                if(i) gpInputOfs << ",";
                gpInputOfs
                    << "\"" << outputName << ".plot" << "\""
                    << " index " << i
                    << "w lp ps 0.4 pt 7 lt "
                    << (color_vec.size()>i ? color_vec[i] : 0)
                    << " title \""
                    << pass_rate_vec[i]*100 << "% of pictures\"";
            }
            gpInputOfs << std::endl;
        }

        template<class T>
        void write_data(std::vector<std::vector<double>> plot_dat, std::string outputPath){

            std::vector<std::string> plot_out(pass_rate_vec.size());

            for(long i = 0; i <= step_cnt<T>; ++i){
                auto d = plot_max<T>() * ((long double)i / step_cnt<T>);
                std::vector<double> rate_vec;
                rate_vec.reserve(plot_dat.size());
                for(auto&& pd : plot_dat){
                    auto ub = std::upper_bound(pd.begin(), pd.end(), d);
                    auto cnt_pass = long(ub - pd.begin());
                    auto rate = double(pd.size() - cnt_pass) / pd.size();
                    rate_vec.push_back(rate);
                }

                std::sort(rate_vec.begin(), rate_vec.end());

                for(int i=0; i<pass_rate_vec.size(); ++i)
                    plot_out[i] += std::to_string(d) + " " + std::to_string(rate_vec.at(rate_vec.size()*pass_rate_vec[i]-1)) + "\n";
            }

            std::ofstream ofs(outputPath + ".plot");
            ofs << "# torelance, error_rate" << "\n\n" << std::endl;
            for(int i=0; i < plot_out.size(); ++i)
                ofs << "# rate" << pass_rate_vec[i] << "\n" << plot_out[i] << "\n\n" << std::endl;
        }
    }
}

