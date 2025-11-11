'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';

export default function DataPage() {
  const [availableData, setAvailableData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  // Form state
  const [symbols, setSymbols] = useState('BTCUSDT,ETHUSDT');
  const [startDate, setStartDate] = useState('2019-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');

  useEffect(() => {
    fetchAvailableData();
  }, []);

  const fetchAvailableData = async () => {
    setLoading(true);
    try {
      const response = await apiClient.getAvailableData();
      setAvailableData(response.data.data || []);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (e: React.FormEvent) => {
    e.preventDefault();
    setDownloading(true);

    try {
      const symbolList = symbols.split(',').map(s => s.trim());
      const response = await apiClient.downloadData({
        symbols: symbolList,
        start_date: startDate,
        end_date: endDate,
        timeframe: '1h',
        exchange: 'binance',
      });

      alert(`Download started! Task ID: ${response.data.task_id}`);

      // Refresh available data after a delay
      setTimeout(fetchAvailableData, 5000);
    } catch (error: any) {
      alert(`Download failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Data Management
      </h1>

      {/* Download Form */}
      <div className="card mb-8">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Download 1H Market Data
        </h2>
        <form onSubmit={handleDownload} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Symbols (comma-separated)
            </label>
            <input
              type="text"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              className="input w-full"
              placeholder="BTCUSDT,ETHUSDT,SOLUSDT"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="input w-full"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                End Date
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="input w-full"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={downloading}
            className="btn btn-primary w-full"
          >
            {downloading ? 'Downloading...' : 'Download Data'}
          </button>
        </form>
      </div>

      {/* Available Data */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Available Data Files
          </h2>
          <button
            onClick={fetchAvailableData}
            disabled={loading}
            className="btn btn-secondary"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {availableData.length === 0 ? (
          <p className="text-gray-600 dark:text-gray-400 text-center py-8">
            No data files available. Download some data to get started.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Filename
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Size
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Modified
                  </th>
                </tr>
              </thead>
              <tbody>
                {availableData.map((file, index) => (
                  <tr
                    key={index}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    <td className="py-3 px-4 text-sm text-gray-900 dark:text-white font-mono">
                      {file.filename}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                      {file.size_mb} MB
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                      {new Date(file.modified).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
