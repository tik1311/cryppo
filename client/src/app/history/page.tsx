'use client';

export default function HistoryPage() {
  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Training History
      </h1>

      <div className="card">
        <div className="text-center py-12">
          <div className="text-6xl mb-4">📊</div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Coming Soon
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Training history and comparison features will be available in the next update.
          </p>
          <div className="space-y-2 text-left max-w-md mx-auto">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              • View all training runs
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              • Compare model performance
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              • Analyze training metrics
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              • Export results
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
