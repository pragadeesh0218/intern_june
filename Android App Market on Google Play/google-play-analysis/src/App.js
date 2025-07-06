import React, { useState, useEffect, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, ScatterChart, Scatter, Legend, Tooltip, Area, AreaChart } from 'recharts';
import { Upload, Download, BarChart3, TrendingUp, Star, DollarSign, Users, Filter, Search, Target, Smartphone, Activity, Award, Zap } from 'lucide-react';
import * as Papa from 'papaparse';
import _ from 'lodash';

const getSentimentColor = (score) => {
  const s = parseFloat(score);
  if (s >= 0.5) return 'text-green-600';
  if (s <= -0.3) return 'text-red-600';
  return 'text-yellow-600';
};

const GooglePlayStoreAnalysis = () => {
  const [rawData, setRawData] = useState([]);
  const [processedData, setProcessedData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedMetric, setSelectedMetric] = useState('Rating');
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState('overview');
  const [reviewData, setReviewData] = useState([]);
  const [sortBy, setSortBy] = useState('rating');
  const [filterType, setFilterType] = useState('All');

  // Enhanced sample data with more realistic metrics
  const sampleData = [
    { App: 'Instagram', Category: 'Social', Rating: 4.5, Reviews: 15000000, Size: 50.2, Installs: 5000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-15', ContentRating: 'Teen' },
    { App: 'WhatsApp', Category: 'Communication', Rating: 4.4, Reviews: 25000000, Size: 25.8, Installs: 5000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-20', ContentRating: 'Everyone' },
    { App: 'Spotify', Category: 'Music', Rating: 4.3, Reviews: 8500000, Size: 82.1, Installs: 1000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-18', ContentRating: 'Teen' },
    { App: 'Netflix', Category: 'Entertainment', Rating: 4.2, Reviews: 12000000, Size: 125.6, Installs: 1000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-22', ContentRating: 'Teen' },
    { App: 'Uber', Category: 'Travel', Rating: 4.1, Reviews: 7500000, Size: 68.4, Installs: 500000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-12', ContentRating: 'Everyone' },
    { App: 'Adobe Photoshop', Category: 'Photography', Rating: 4.0, Reviews: 1200000, Size: 245.7, Installs: 100000000, Price: 9.99, Type: 'Paid', LastUpdated: '2024-01-10', ContentRating: 'Everyone' },
    { App: 'Candy Crush Saga', Category: 'Game', Rating: 4.4, Reviews: 18000000, Size: 95.3, Installs: 2000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-25', ContentRating: 'Everyone' },
    { App: 'Microsoft Office', Category: 'Productivity', Rating: 4.3, Reviews: 3200000, Size: 312.8, Installs: 500000000, Price: 6.99, Type: 'Paid', LastUpdated: '2024-01-14', ContentRating: 'Everyone' },
    { App: 'TikTok', Category: 'Social', Rating: 4.2, Reviews: 35000000, Size: 78.9, Installs: 3000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-28', ContentRating: 'Teen' },
    { App: 'Google Maps', Category: 'Maps', Rating: 4.6, Reviews: 22000000, Size: 42.1, Installs: 5000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-30', ContentRating: 'Everyone' },
    { App: 'YouTube', Category: 'Entertainment', Rating: 4.1, Reviews: 28000000, Size: 156.2, Installs: 5000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-26', ContentRating: 'Teen' },
    { App: 'Facebook', Category: 'Social', Rating: 3.9, Reviews: 45000000, Size: 89.5, Installs: 5000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-19', ContentRating: 'Teen' },
    { App: 'Zoom', Category: 'Communication', Rating: 4.2, Reviews: 5500000, Size: 45.7, Installs: 500000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-17', ContentRating: 'Everyone' },
    { App: 'Minecraft', Category: 'Game', Rating: 4.5, Reviews: 8200000, Size: 142.3, Installs: 100000000, Price: 6.99, Type: 'Paid', LastUpdated: '2024-01-21', ContentRating: 'Everyone 10+' },
    { App: 'Amazon', Category: 'Shopping', Rating: 4.0, Reviews: 12500000, Size: 67.8, Installs: 1000000000, Price: 0, Type: 'Free', LastUpdated: '2024-01-23', ContentRating: 'Everyone' }
  ];

  const sampleReviewData = [
    { App: 'Instagram', Sentiment: 'Positive' },
    { App: 'Instagram', Sentiment: 'Positive' },
    { App: 'Instagram', Sentiment: 'Negative' },
    { App: 'WhatsApp', Sentiment: 'Positive' },
    { App: 'WhatsApp', Sentiment: 'Positive' },
    { App: 'Spotify', Sentiment: 'Neutral' },
    { App: 'Netflix', Sentiment: 'Positive' },
    { App: 'Uber', Sentiment: 'Negative' },
    { App: 'Adobe Photoshop', Sentiment: 'Positive' },
    { App: 'Candy Crush Saga', Sentiment: 'Positive' },
    { App: 'TikTok', Sentiment: 'Positive' },
    { App: 'TikTok', Sentiment: 'Negative' },
    { App: 'Google Maps', Sentiment: 'Positive' },
    { App: 'YouTube', Sentiment: 'Neutral' },
    { App: 'Facebook', Sentiment: 'Negative' }
  ];

  useEffect(() => {
    setRawData(sampleData);
    setProcessedData(sampleData);
    setReviewData(sampleReviewData);
  }, []);

  const sentimentSummary = useMemo(() => {
    const grouped = _.groupBy(reviewData, 'App');
    return _.mapValues(grouped, reviews => {
      const positive = reviews.filter(r => r.Sentiment === 'Positive').length;
      const negative = reviews.filter(r => r.Sentiment === 'Negative').length;
      const neutral = reviews.filter(r => r.Sentiment === 'Neutral').length;
      const total = reviews.length;
      return {
        positive,
        negative,
        neutral,
        total,
        score: total > 0 ? ((positive - negative) / total).toFixed(2) : "0.00"
      };
    });
  }, [reviewData]);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setLoading(true);
      Papa.parse(file, {
        complete: (results) => {
          if (results.data && results.data.length > 0) {
            setRawData(results.data);
            setProcessedData(results.data);
          }
          setLoading(false);
        },
        header: true,
        skipEmptyLines: true,
        dynamicTyping: true,
        transformHeader: (header) => header.trim()
      });
    }
  };

  const cleanData = useMemo(() => {
    return processedData.map(app => ({
      ...app,
      Rating: parseFloat(app.Rating) || 0,
      Reviews: parseInt(String(app.Reviews).replace(/,/g, '')) || 0,
      Size: app.Size && typeof app.Size === 'string' ? parseFloat(app.Size.replace(/[^0-9.]/g, '')) || 0 : typeof app.Size === 'number' ? app.Size : 0,
      Installs: app.Installs ? parseInt(String(app.Installs).replace(/[^0-9]/g, '')) || 0 : 0,
      Price: app.Price ? parseFloat(String(app.Price).replace(/[^0-9.]/g, '')) || 0 : 0,
      Category: app.Category || 'Unknown',
      Type: app.Type || 'Free',
      ContentRating: app.ContentRating || 'Everyone'
    }));
  }, [processedData]);

  const filteredData = useMemo(() => {
    let filtered = cleanData;
    
    if (selectedCategory !== 'All') {
      filtered = filtered.filter(app => app.Category === selectedCategory);
    }
    
    if (filterType !== 'All') {
      filtered = filtered.filter(app => app.Type === filterType);
    }
    
    if (searchTerm) {
      filtered = filtered.filter(app => 
        app.App && app.App.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    return filtered;
  }, [cleanData, selectedCategory, filterType, searchTerm]);

  const categories = useMemo(() => {
    const cats = _.uniq(cleanData.map(app => app.Category)).filter(Boolean);
    return ['All', ...cats.sort()];
  }, [cleanData]);

  const categoryDistribution = useMemo(() => {
    const grouped = _.groupBy(cleanData, 'Category');
    return Object.entries(grouped).map(([category, apps]) => ({
      category,
      count: apps.length,
      avgRating: _.meanBy(apps, 'Rating').toFixed(1),
      totalInstalls: _.sumBy(apps, 'Installs'),
      avgSize: _.meanBy(apps, 'Size').toFixed(1)
    })).sort((a, b) => b.count - a.count);
  }, [cleanData]);

  const ratingDistribution = useMemo(() => {
    const ratings = cleanData.map(app => Math.floor(app.Rating));
    const grouped = _.groupBy(ratings);
    return Object.entries(grouped).map(([rating, apps]) => ({
      rating: `${rating}.0+`,
      count: apps.length,
      percentage: ((apps.length / cleanData.length) * 100).toFixed(1)
    })).sort((a, b) => parseFloat(a.rating) - parseFloat(b.rating));
  }, [cleanData]);

  const typeDistribution = useMemo(() => {
    const grouped = _.groupBy(cleanData, 'Type');
    return Object.entries(grouped).map(([type, apps]) => ({
      type,
      count: apps.length,
      percentage: ((apps.length / cleanData.length) * 100).toFixed(1),
      avgRating: _.meanBy(apps, 'Rating').toFixed(1)
    }));
  }, [cleanData]);

  const sizeRatingData = useMemo(() => {
    return cleanData
      .filter(app => app.Size > 0 && app.Rating > 0)
      .map(app => ({
        size: app.Size,
        rating: app.Rating,
        category: app.Category,
        name: app.App,
        installs: app.Installs
      }));
  }, [cleanData]);

  const topApps = useMemo(() => {
    const sortedApps = [...cleanData].sort((a, b) => {
      switch (sortBy) {
        case 'rating': return b.Rating - a.Rating;
        case 'reviews': return b.Reviews - a.Reviews;
        case 'installs': return b.Installs - a.Installs;
        case 'size': return b.Size - a.Size;
        default: return b.Rating - a.Rating;
      }
    });
    
    return sortedApps.slice(0, 10).map(app => ({
      ...app,
      sentiment: sentimentSummary[app.App] || {}
    }));
  }, [cleanData, sortBy, sentimentSummary]);

  const installsVsRatingData = useMemo(() => {
    const sizeCategories = ['0-50MB', '50-100MB', '100-200MB', '200MB+'];
    
    return sizeCategories.map(category => {
      let apps = [];
      
      switch (category) {
        case '0-50MB':
          apps = cleanData.filter(app => app.Size >= 0 && app.Size < 50);
          break;
        case '50-100MB':
          apps = cleanData.filter(app => app.Size >= 50 && app.Size < 100);
          break;
        case '100-200MB':
          apps = cleanData.filter(app => app.Size >= 100 && app.Size < 200);
          break;
        case '200MB+':
          apps = cleanData.filter(app => app.Size >= 200);
          break;
      }
      
      return {
        category,
        count: apps.length,
        avgRating: apps.length > 0 ? _.meanBy(apps, 'Rating').toFixed(1) : 0,
        totalInstalls: _.sumBy(apps, 'Installs')
      };
    }).filter(item => item.count > 0);
  }, [cleanData]);

  const statistics = useMemo(() => {
    const validRatings = cleanData.filter(app => app.Rating > 0);
    const validSizes = cleanData.filter(app => app.Size > 0);
    const paidApps = cleanData.filter(app => app.Type === 'Paid');
    const totalInstalls = _.sumBy(cleanData, 'Installs');
    
    return {
      totalApps: cleanData.length,
      avgRating: validRatings.length > 0 ? _.meanBy(validRatings, 'Rating').toFixed(2) : 0,
      avgSize: validSizes.length > 0 ? _.meanBy(validSizes, 'Size').toFixed(1) : 0,
      freeApps: cleanData.filter(app => app.Type === 'Free').length,
      paidApps: paidApps.length,
      avgPrice: paidApps.length > 0 ? _.meanBy(paidApps, 'Price').toFixed(2) : 0,
      topCategory: categoryDistribution.length > 0 ? categoryDistribution[0].category : 'N/A',
      totalInstalls,
      avgReviews: cleanData.length > 0 ? _.meanBy(cleanData, 'Reviews').toFixed(0) : 0
    };
  }, [cleanData, categoryDistribution]);

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#F97316', '#84CC16'];

  const TabButton = ({ id, label, icon: Icon, active, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
        active 
          ? 'bg-blue-500 text-white shadow-lg transform scale-105' 
          : 'bg-white text-gray-600 hover:bg-gray-50 hover:shadow-md border border-gray-200'
      }`}
    >
      <Icon size={16} />
      {label}
    </button>
  );

  const StatCard = ({ title, value, icon: Icon, color, trend }) => (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition-shadow duration-300">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-600 text-sm font-medium">{title}</p>
          <p className={`text-2xl font-bold ${color}`}>{value}</p>
          {trend && <p className="text-xs text-gray-500 mt-1">{trend}</p>}
        </div>
        <div className={`p-3 rounded-full ${color.replace('text-', 'bg-').replace('-600', '-100')}`}>
          <Icon className={color.replace('-600', '-500')} size={24} />
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Google Play Store Analytics
          </h1>
          <p className="text-gray-600 text-lg">
            Comprehensive market analysis with advanced insights and interactive visualizations
          </p>
        </div>

        {/* File Upload */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Upload size={20} />
            Data Management
          </h2>
          <div className="flex items-center gap-4 flex-wrap">
            <label className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg cursor-pointer hover:from-blue-600 hover:to-blue-700 transition-all shadow-lg">
              <Upload size={16} />
              Upload CSV File
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
            <span className="text-gray-600 font-medium">
              {loading ? (
                <span className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                  Processing...
                </span>
              ) : (
                `📊 ${cleanData.length} apps loaded`
              )}
            </span>
          </div>
        </div>

        {/* Enhanced Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Total Apps"
            value={statistics.totalApps.toLocaleString()}
            icon={Smartphone}
            color="text-blue-600"
            trend="In dataset"
          />
          <StatCard
            title="Average Rating"
            value={`${statistics.avgRating}⭐`}
            icon={Star}
            color="text-yellow-600"
            trend="Out of 5.0"
          />
          <StatCard
            title="Total Installs"
            value={`${(statistics.totalInstalls / 1000000000).toFixed(1)}B`}
            icon={Activity}
            color="text-green-600"
            trend="Combined downloads"
          />
          <StatCard
            title="Free Apps"
            value={`${((statistics.freeApps / statistics.totalApps) * 100).toFixed(1)}%`}
            icon={Users}
            color="text-purple-600"
            trend={`${statistics.freeApps} of ${statistics.totalApps}`}
          />
        </div>

        {/* Enhanced Controls */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-200">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex items-center gap-2">
              <Filter size={16} className="text-gray-600" />
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {categories.map(cat => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <Target size={16} className="text-gray-600" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="All">All Types</option>
                <option value="Free">Free</option>
                <option value="Paid">Paid</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <Search size={16} className="text-gray-600" />
              <input
                type="text"
                placeholder="Search apps..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div className="flex items-center gap-2">
              <Award size={16} className="text-gray-600" />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="rating">Sort by Rating</option>
                <option value="reviews">Sort by Reviews</option>
                <option value="installs">Sort by Installs</option>
                <option value="size">Sort by Size</option>
              </select>
            </div>
          </div>
        </div>

        {/* Enhanced Tabs */}
        <div className="flex gap-2 mb-8 overflow-x-auto">
          <TabButton
            id="overview"
            label="Overview"
            icon={BarChart3}
            active={activeTab === 'overview'}
            onClick={setActiveTab}
          />
          <TabButton
            id="categories"
            label="Categories"
            icon={Target}
            active={activeTab === 'categories'}
            onClick={setActiveTab}
          />
          <TabButton
            id="ratings"
            label="Ratings"
            icon={Star}
            active={activeTab === 'ratings'}
            onClick={setActiveTab}
          />
          <TabButton
            id="analysis"
            label="Analysis"
            icon={TrendingUp}
            active={activeTab === 'analysis'}
            onClick={setActiveTab}
          />
          <TabButton
            id="insights"
            label="Insights"
            icon={Zap}
            active={activeTab === 'insights'}
            onClick={setActiveTab}
          />
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Users size={20} />
                App Type Distribution
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={typeDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ type, percentage }) => `${type} (${percentage}%)`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {typeDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [value, 'Apps']} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart3 size={20} />
                Top Categories
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={categoryDistribution.slice(0, 8)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity size={20} />
                App Size Distribution
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={installsVsRatingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip />
                  <Area type="monotone" dataKey="count" stroke="#10B981" fill="#10B981" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Award size={20} />
                Top Performing Apps
              </h3>
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {topApps.slice(0, 5).map((app, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                        {index + 1}
                      </div>
                      <div>
                        <p className="font-medium">{app.App}</p>
                        <p className="text-sm text-gray-600">{app.Category}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Star className="text-yellow-500" size={16} />
                      <span className="font-semibold">{app.Rating}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'categories' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Category Distribution</h3>
              <ResponsiveContainer width="100%" height={400}>
                <PieChart>
                  <Pie
                    data={categoryDistribution.slice(0, 8)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ category, count }) => `${category} (${count})`}
                    outerRadius={120}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {categoryDistribution.slice(0, 8).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Average Rating by Category</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={categoryDistribution.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" angle={-45} textAnchor="end" height={80} />
                  <YAxis domain={[0, 5]} />
                  <Tooltip />
                  <Bar dataKey="avgRating" fill="#10B981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'ratings' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Rating Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={ratingDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="rating" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Top Rated Apps</h3>
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {topApps.map((app, index) => (
                  <div
                    key={index}
                    className="flex flex-col gap-1 p-3 bg-gray-50 rounded-lg border border-gray-200"
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium">{app.App}</p>
                        <p className="text-sm text-gray-600">{app.Category}</p>
                      </div>
                      <div className="flex items-center gap-1">
                        <Star className="text-yellow-500" size={16} />
                        <span className="font-semibold">{app.Rating}</span>
                      </div>
                    </div>

                    <div className="flex justify-between text-sm text-gray-600 mt-1">
                      <p>👍 {app.sentiment?.positive || 0}</p>
                      <p>😐 {app.sentiment?.neutral || 0}</p>
                      <p>👎 {app.sentiment?.negative || 0}</p>
                      <p className={`font-medium ${getSentimentColor(app.sentiment?.score)}`}>
                        Score: {app.sentiment?.score || "0.00"}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="grid grid-cols-1 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">App Size vs Rating Correlation</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="size" 
                    type="number" 
                    name="Size (MB)"
                    label={{ value: 'Size (MB)', position: 'bottom' }}
                  />
                  <YAxis 
                    dataKey="rating" 
                    type="number" 
                    name="Rating"
                    domain={[0, 5]}
                    label={{ value: 'Rating', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value, name) => [value, name === 'rating' ? 'Rating' : 'Size (MB)']}
                    labelFormatter={(label) => `App: ${label}`}
                  />
                  <Scatter 
                    data={sizeRatingData.slice(0, 100)} 
                    fill="#8884d8" 
                    name="Apps"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">App Size Categories Analysis</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={installsVsRatingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="avgRating" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'insights' && (
          <div className="grid grid-cols-1 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Market Insights</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <h4 className="font-semibold text-blue-800 mb-2">📊 Market Leader</h4>
                  <p className="text-blue-700">
                    {statistics.topCategory} dominates with {categoryDistribution[0]?.count || 0} apps, 
                    representing {((categoryDistribution[0]?.count || 0) / statistics.totalApps * 100).toFixed(1)}% of the market
                  </p>
                </div>
                <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                  <h4 className="font-semibold text-green-800 mb-2">⭐ Quality Standard</h4>
                  <p className="text-green-700">
                    Average rating of {statistics.avgRating}/5.0 indicates strong user satisfaction across the platform
                  </p>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <h4 className="font-semibold text-purple-800 mb-2">💰 Business Model</h4>
                  <p className="text-purple-700">
                    {((statistics.freeApps / statistics.totalApps) * 100).toFixed(1)}% free apps suggest 
                    freemium and ad-supported models dominate
                  </p>
                </div>
                <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                  <h4 className="font-semibold text-orange-800 mb-2">📱 Size Optimization</h4>
                  <p className="text-orange-700">
                    Average size of {statistics.avgSize}MB shows developers optimize for mobile constraints
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg">
                  <div className="text-2xl font-bold">{(statistics.totalInstalls / 1000000000).toFixed(1)}B</div>
                  <div className="text-sm opacity-90">Total Downloads</div>
                </div>
                <div className="text-center p-4 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg">
                  <div className="text-2xl font-bold">{Math.round(statistics.avgReviews / 1000)}K</div>
                  <div className="text-sm opacity-90">Avg Reviews</div>
                </div>
                <div className="text-center p-4 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-lg">
                  <div className="text-2xl font-bold">${statistics.avgPrice}</div>
                  <div className="text-sm opacity-90">Avg Paid App Price</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 p-6 bg-white rounded-xl shadow-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold mb-2">Export & Share</h3>
              <p className="text-gray-600">
                Generate comprehensive reports and export your analysis results
              </p>
            </div>
            <button className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:from-green-600 hover:to-green-700 transition-all shadow-lg">
              <Download size={16} />
              Export Results
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GooglePlayStoreAnalysis;