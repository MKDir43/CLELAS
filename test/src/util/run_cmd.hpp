#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <tuple>
#include <exception>

enum RunCmd_RunOption_WithStdOut{ with_stdout = 1 };

namespace RunCmd_internal{
    template<bool withStdout>
    inline std::tuple<int, std::string> RunCmd_impl(const std::string command){
        std::string output;
        constexpr int buf_size = 1024;
        char buf[buf_size + 4];
        FILE *ep;
        if(!(ep = popen(command.data(), "r")))
            throw std::runtime_error("could not open: " + command);
        while(fgets(buf, buf_size, ep))
            if(withStdout)
                output += buf;
        int raw_return_code = pclose(ep);
        int return_code = WEXITSTATUS(raw_return_code);
        return std::tuple<int, std::string>{
            return_code,
            std::move(output)
        };
    }
}

inline std::tuple<int, std::string> RunCmd(const std::string command, RunCmd_RunOption_WithStdOut){
    return RunCmd_internal::RunCmd_impl<true>(std::move(command));
}

inline int RunCmd(const std::string command){
    int ret;
    std::tie(ret, std::ignore) = RunCmd_internal::RunCmd_impl<false>(std::move(command));
    return ret;
}
