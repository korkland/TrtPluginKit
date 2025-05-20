#pragma once

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <string>

// Defines the getCreators() function for a single plugin creator
#define DEFINE_TRT_PLUGIN_CREATOR(PluginCreatorClass)                             \
extern "C" TENSORRTAPI nvinfer1::IPluginCreatorInterface* const* getCreators(     \
    int32_t& nbCreators)                                                          \
{                                                                                 \
    nbCreators = 1;                                                               \
    static PluginCreatorClass creatorInstance;                                    \
    static nvinfer1::IPluginCreatorInterface* const creatorList[] = {             \
        &creatorInstance                                                          \
    };                                                                            \
    return creatorList;                                                           \
}

namespace nvinfer1
{
namespace plugin
{

class IPluginBase : public IPluginV3,
                    public IPluginV3OneCore
{
public:
    IPluginBase(const std::string& pluginName, const std::string& pluginVersion = "1", const std::string& pluginNamespace = "")
        : m_pluginName(pluginName)
        , m_pluginVersion(pluginVersion)
        , m_pluginNamespace(pluginNamespace)
    {}

    char const* getPluginName() const noexcept override
    {
        return m_pluginName.c_str();
    }
    char const* getPluginVersion() const noexcept override
    {
        return m_pluginVersion.c_str();
    }
    char const* getPluginNamespace() const noexcept override
    {
        return m_pluginNamespace.c_str();
    }

protected:
    std::string m_pluginName;;
    std::string m_pluginVersion;
    std::string m_pluginNamespace;
};

class IPluginCreatorBase : public IPluginCreatorV3One
{
public:
    IPluginCreatorBase(const std::string& pluginName, const std::string& pluginVersion = "1", const std::string& pluginNamespace = "")
        : m_pluginName(pluginName)
        , m_pluginVersion(pluginVersion)
        , m_pluginNamespace(pluginNamespace)
    {}

    char const* getPluginName() const noexcept override
    {
        return m_pluginName.c_str();
    }
    char const* getPluginVersion() const noexcept override
    {
        return m_pluginVersion.c_str();
    }
    char const* getPluginNamespace() const noexcept override
    {
        return m_pluginNamespace.c_str();
    }

protected:
    std::string m_pluginName;;
    std::string m_pluginVersion;
    std::string m_pluginNamespace;
};

} // namespace plugin
} // namespace nvinfer1