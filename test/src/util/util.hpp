#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

namespace Util{
    namespace fs = boost::filesystem;

    inline std::vector<std::string> getFileList(std::string path, std::string ext = ".png") {
        std::vector<std::string> fileNames;
        const fs::path dirPath(path);
        for(const auto& e: boost::make_iterator_range(fs::directory_iterator(dirPath))) {
            std::string entryName = e.path().filename().string();
            std::string entryExt = e.path().extension().string();
            if(entryExt == ext)
                fileNames.push_back(entryName);
        }
        std::sort(fileNames.begin(), fileNames.end());
        return fileNames;
    }

    inline std::string pathCat(std::string dirPath, std::string fName) {
        fs::path dir(dirPath);
        fs::path file(fName);
        auto full_path = dir / file;
        return full_path.string();
    }

}
