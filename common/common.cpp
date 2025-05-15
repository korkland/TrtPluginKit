#include "common.h"
#include <mutex>

class ThreadSafeLoggerFinder
{
public:
    ThreadSafeLoggerFinder() = default;

    //! Set the logger finder.
    void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (m_loggerFinder == nullptr && finder != nullptr)
        {
            m_loggerFinder = finder;
        }
    }

    //! Get the logger.
    nvinfer1::ILogger* getLogger() noexcept
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (m_loggerFinder != nullptr)
        {
            return m_loggerFinder->findLogger();
        }
        return nullptr;
    }

private:
    nvinfer1::ILoggerFinder* m_loggerFinder{nullptr};
    std::mutex m_mutex;
};

ThreadSafeLoggerFinder gLoggerFinder;


extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    gLoggerFinder.setLoggerFinder(finder);
}