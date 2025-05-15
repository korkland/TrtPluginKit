#pragma once

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <string>

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