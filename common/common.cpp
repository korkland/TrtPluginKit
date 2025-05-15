#include "common.h"
#include <mutex>

namespace nvinfer1
{
namespace plugin
{

class ThreadSafeLoggerFinder
{
private:
    ILoggerFinder* m_loggerFinder{nullptr};
    std::mutex m_mutex;

public:
    ThreadSafeLoggerFinder() = default;

    //! Set the logger finder.
    void setLoggerFinder(ILoggerFinder* finder)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (m_loggerFinder == nullptr && finder != nullptr)
        {
            m_loggerFinder = finder;
        }
    }

    //! Get the logger.
    ILogger* getLogger() noexcept
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (m_loggerFinder != nullptr)
        {
            return m_loggerFinder->findLogger();
        }
        return nullptr;
    }
};

ThreadSafeLoggerFinder gLoggerFinder;

} // namespace plugin
} // namespace nvinfer1

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    nvinfer1::plugin::gLoggerFinder.setLoggerFinder(finder);
}