#pragma once

#include <sstream>
#include <string>
#include <exception>
#include <cstdlib>

#include <NvInferRuntime.h>

namespace nvinfer1
{
namespace plugin
{

inline void caughtError(std::exception const& e)
{
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());
}

inline void logInfo(char const* msg)
{
    getLogger()->log(nvinfer1::ILogger::Severity::kINFO, msg);
}

#define PLUGIN_ASSERT(val) reportAssertion((val), #val, __FILE__, __LINE__)
inline void reportAssertion(bool success, char const* msg, char const* file, int32_t line)
{
    if (!success)
    {
        std::ostringstream stream;
        stream << "Assertion failed: " << msg << std::endl
               << file << ':' << line << std::endl
               << "Aborting..." << std::endl;
        getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
        std::abort();
    }
}

#define PLUGIN_VALIDATE(val) reportValidation((val), #val, __FILE__, __LINE__)
inline void reportValidation(bool success, char const* msg, char const* file, int32_t line)
{
    if (!success)
    {
        std::ostringstream stream;
        stream << "Validation failed: " << msg << std::endl
               << file << ':' << line << std::endl;
        getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    }
}

} // namespace plugin
} // namespace nvinfer1

// for plugin registration
extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder);
extern "C" TENSORRTAPI nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);
