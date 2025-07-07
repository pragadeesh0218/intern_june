import React, { useState, useEffect, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Shield, TrendingUp, Activity, Database, Brain, Eye, Zap } from 'lucide-react';

const FraudDetectionSystem = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedModel, setSelectedModel] = useState('logistic');
  const [threshold, setThreshold] = useState(0.5);
  const [realTimeData, setRealTimeData] = useState([]);
  const [isMonitoring, setIsMonitoring] = useState(false);

  // Generate data based on Kaggle Credit Card Fraud Dataset structure
  const generateKaggleStyleData = (count = 1000) => {
    const data = [];
    for (let i = 0; i < count; i++) {
      const isFraud = Math.random() < 0.00172; // Actual fraud rate: 0.172%
      
      // Time: seconds elapsed since first transaction (0 to 172,792 seconds ~48 hours)
      const time = Math.floor(Math.random() * 172792);
      
      // Amount: Transaction amount (real dataset: 0 to 25,691.16)
      const amount = isFraud 
        ? Math.random() * 2000 + 1 // Fraudulent amounts tend to be varied
        : Math.random() * 500 + 0.01; // Normal amounts are typically smaller
      
      // V1-V28: PCA components (anonymized features)
      const pcaFeatures = {};
      for (let j = 1; j <= 28; j++) {
        // Generate realistic PCA values (typically between -3 and 3)
        pcaFeatures[`V${j}`] = (Math.random() - 0.5) * 6;
      }
      
      // Add some correlation for fraud patterns
      if (isFraud) {
        pcaFeatures.V1 = Math.random() * 2 - 1; // Slightly different distribution for fraud
        pcaFeatures.V2 = Math.random() * 3 - 1.5;
        pcaFeatures.V3 = Math.random() * 4 - 2;
        pcaFeatures.V4 = Math.random() * 3 - 1.5;
        pcaFeatures.V17 = Math.random() * -2 - 1; // Negative correlation example
      }
      
      data.push({
        id: i + 1,
        Time: time,
        Amount: parseFloat(amount.toFixed(2)),
        Class: isFraud ? 1 : 0, // 0 = Normal, 1 = Fraud
        ...pcaFeatures,
        // Derived features for visualization
        timeOfDay: (time % 86400) / 3600, // Convert to hours of day
        dayOfWeek: Math.floor(time / 86400), // Which day (0-1 for 48 hours)
        isFraud: isFraud,
        riskScore: isFraud ? Math.random() * 0.4 + 0.6 : Math.random() * 0.4,
        predicted: Math.random() > 0.08 ? isFraud : !isFraud // 92% accuracy (realistic for this dataset)
      });
    }
    return data;
  };

  const [transactionData] = useState(() => generateKaggleStyleData(10000)); // Increased to 10k for more realistic dataset size

  // Model performance metrics (based on actual Kaggle dataset benchmarks)
  const modelMetrics = {
    logistic: { accuracy: 0.9991, precision: 0.88, recall: 0.56, f1Score: 0.69 },
    decisionTree: { accuracy: 0.9984, precision: 0.73, recall: 0.76, f1Score: 0.75 },
    randomForest: { accuracy: 0.9995, precision: 0.95, recall: 0.75, f1Score: 0.84 },
    neuralNetwork: { accuracy: 0.9994, precision: 0.92, recall: 0.79, f1Score: 0.85 },
    svm: { accuracy: 0.9993, precision: 0.90, recall: 0.73, f1Score: 0.81 }
  };

  // Real-time monitoring simulation
  useEffect(() => {
    let interval;
    if (isMonitoring) {
      interval = setInterval(() => {
        const newTransaction = generateKaggleStyleData(1)[0];
        newTransaction.timestamp = new Date().toLocaleTimeString();
        setRealTimeData(prev => [...prev.slice(-9), newTransaction]);
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isMonitoring]);

  // Analytics calculations
  const analytics = useMemo(() => {
    const totalTransactions = transactionData.length;
    const fraudulentTransactions = transactionData.filter(t => t.isFraud).length;
    const detectedFraud = transactionData.filter(t => t.isFraud && t.predicted).length;
    const falsePositives = transactionData.filter(t => !t.isFraud && t.predicted).length;
    
    return {
      totalTransactions,
      fraudulentTransactions,
      fraudRate: (fraudulentTransactions / totalTransactions * 100).toFixed(2),
      detectionRate: (detectedFraud / fraudulentTransactions * 100).toFixed(2),
      falsePositiveRate: (falsePositives / (totalTransactions - fraudulentTransactions) * 100).toFixed(2),
      avgTransactionAmount: (transactionData.reduce((sum, t) => sum + t.amount, 0) / totalTransactions).toFixed(2)
    };
  }, [transactionData]);

  // Anomaly detection data - using actual Kaggle dataset structure
  const anomalyData = transactionData.slice(0, 500).map(t => ({
    amount: t.Amount,
    time: t.Time,
    v1: t.V1,
    v2: t.V2,
    v17: t.V17, // V17 is often important for fraud detection
    isFraud: t.isFraud,
    anomalyScore: t.riskScore
  }));

  // Feature importance data (based on actual Kaggle dataset analysis)
  const featureImportance = [
    { feature: 'V14', importance: 0.12 },
    { feature: 'V4', importance: 0.11 },
    { feature: 'V11', importance: 0.10 },
    { feature: 'V2', importance: 0.09 },
    { feature: 'V19', importance: 0.08 },
    { feature: 'V21', importance: 0.07 },
    { feature: 'V27', importance: 0.06 },
    { feature: 'V20', importance: 0.06 },
    { feature: 'V16', importance: 0.05 },
    { feature: 'V7', importance: 0.05 },
    { feature: 'V12', importance: 0.05 },
    { feature: 'V18', importance: 0.04 },
    { feature: 'Amount', importance: 0.04 },
    { feature: 'V10', importance: 0.04 },
    { feature: 'Time', importance: 0.04 }
  ];

  const TabButton = ({ id, label, icon: Icon }) => (
    <button
      onClick={() => setActiveTab(id)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '8px 16px',
        borderRadius: '8px',
        transition: 'all 0.2s',
        backgroundColor: activeTab === id ? '#2563eb' : '#f3f4f6',
        color: activeTab === id ? 'white' : '#374151',
        border: 'none',
        cursor: 'pointer'
      }}
      onMouseEnter={(e) => {
        if (activeTab !== id) {
          e.target.style.backgroundColor = '#e5e7eb';
        }
      }}
      onMouseLeave={(e) => {
        if (activeTab !== id) {
          e.target.style.backgroundColor = '#f3f4f6';
        }
      }}
    >
      <Icon size={16} />
      {label}
    </button>
  );

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '24px', backgroundColor: '#f9fafb', minHeight: '100vh' }}>
      <div style={{ backgroundColor: 'white', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', padding: '32px', marginBottom: '24px' }}>
        <h1 style={{ fontSize: '30px', fontWeight: 'bold', color: '#1f2937', marginBottom: '8px' }}>Advanced Fraud Detection System</h1>
        <p style={{ color: '#6b7280', marginBottom: '24px' }}>
          Comprehensive fraud detection using machine learning, anomaly detection, and real-time monitoring
        </p>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '24px' }}>
          <TabButton id="overview" label="Overview" icon={Activity} />
          <TabButton id="models" label="ML Models" icon={Brain} />
          <TabButton id="anomaly" label="Anomaly Detection" icon={Eye} />
          <TabButton id="monitoring" label="Real-time Monitoring" icon={Zap} />
          <TabButton id="features" label="Feature Engineering" icon={Database} />
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
              <div style={{ backgroundColor: '#eff6ff', padding: '16px', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <Database style={{ color: '#2563eb' }} size={20} />
                  <h3 style={{ fontWeight: '600', color: '#1e3a8a' }}>Total Transactions</h3>
                </div>
                <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#1d4ed8' }}>{analytics.totalTransactions.toLocaleString()}</p>
              </div>
              
              <div style={{ backgroundColor: '#fef2f2', padding: '16px', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <AlertTriangle style={{ color: '#dc2626' }} size={20} />
                  <h3 style={{ fontWeight: '600', color: '#7f1d1d' }}>Fraud Rate</h3>
                </div>
                <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#b91c1c' }}>{analytics.fraudRate}%</p>
              </div>
              
              <div style={{ backgroundColor: '#f0fdf4', padding: '16px', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <Shield style={{ color: '#16a34a' }} size={20} />
                  <h3 style={{ fontWeight: '600', color: '#14532d' }}>Detection Rate</h3>
                </div>
                <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#15803d' }}>{analytics.detectionRate}%</p>
              </div>
              
              <div style={{ backgroundColor: '#fffbeb', padding: '16px', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <TrendingUp style={{ color: '#d97706' }} size={20} />
                  <h3 style={{ fontWeight: '600', color: '#78350f' }}>Avg Transaction</h3>
                </div>
                <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#c2410c' }}>${analytics.avgTransactionAmount}</p>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px' }}>
              <div style={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '8px', padding: '16px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px' }}>Transaction Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={[
                    { name: 'Legitimate', value: analytics.totalTransactions - analytics.fraudulentTransactions, fill: '#10B981' },
                    { name: 'Fraudulent', value: analytics.fraudulentTransactions, fill: '#EF4444' }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white border rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-4">Fraud by Merchant Type</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={Object.entries(
                        transactionData
                          .filter(t => t.isFraud)
                          .reduce((acc, t) => {
                            acc[t.merchantType] = (acc[t.merchantType] || 0) + 1;
                            return acc;
                          }, {})
                      ).map(([type, count]) => ({ name: type, value: count }))}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      label
                    >
                      {['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'].map((color, index) => (
                        <Cell key={`cell-${index}`} fill={color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* ML Models Tab */}
        {activeTab === 'models' && (
          <div className="space-y-6">
            <div className="flex items-center gap-4 mb-6">
              <label className="font-medium">Select Model:</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="px-3 py-2 border rounded-lg"
              >
                <option value="logistic">Logistic Regression</option>
                <option value="decisionTree">Decision Tree</option>
                <option value="randomForest">Random Forest</option>
                <option value="neuralNetwork">Neural Network</option>
                <option value="svm">Support Vector Machine</option>
              </select>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(modelMetrics[selectedModel]).map(([metric, value]) => (
                <div key={metric} className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-gray-800 capitalize">{metric.replace(/([A-Z])/g, ' $1')}</h3>
                  <p className="text-2xl font-bold text-blue-700">{(value * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>

            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Model Performance Comparison</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={Object.entries(modelMetrics).map(([model, metrics]) => ({
                  model: model.replace(/([A-Z])/g, ' $1'),
                  accuracy: metrics.accuracy * 100,
                  precision: metrics.precision * 100,
                  recall: metrics.recall * 100,
                  f1Score: metrics.f1Score * 100
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#8884d8" />
                  <Bar dataKey="precision" fill="#82ca9d" />
                  <Bar dataKey="recall" fill="#ffc658" />
                  <Bar dataKey="f1Score" fill="#ff7300" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Threshold Adjustment</h3>
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">
                  Classification Threshold: {threshold}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.1"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <p className="text-sm text-gray-600">
                Adjusting the threshold affects the trade-off between false positives and false negatives.
                Lower thresholds catch more fraud but increase false alarms.
              </p>
            </div>
          </div>
        )}

        {/* Anomaly Detection Tab */}
        {activeTab === 'anomaly' && (
          <div className="space-y-6">
            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Anomaly Detection Scatter Plot</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={anomalyData.slice(0, 200)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="amount" name="Amount" />
                  <YAxis dataKey="timeOfDay" name="Time of Day" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter
                    name="Legitimate"
                    data={anomalyData.filter(d => !d.isFraud).slice(0, 150)}
                    fill="#10B981"
                  />
                  <Scatter
                    name="Fraudulent"
                    data={anomalyData.filter(d => d.isFraud).slice(0, 50)}
                    fill="#EF4444"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white border rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-4">Risk Score Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={[
                    { range: '0.0-0.2', count: transactionData.filter(t => t.riskScore < 0.2).length },
                    { range: '0.2-0.4', count: transactionData.filter(t => t.riskScore >= 0.2 && t.riskScore < 0.4).length },
                    { range: '0.4-0.6', count: transactionData.filter(t => t.riskScore >= 0.4 && t.riskScore < 0.6).length },
                    { range: '0.6-0.8', count: transactionData.filter(t => t.riskScore >= 0.6 && t.riskScore < 0.8).length },
                    { range: '0.8-1.0', count: transactionData.filter(t => t.riskScore >= 0.8).length }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white border rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-4">Anomaly Detection Techniques</h3>
                <div className="space-y-3">
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <h4 className="font-medium text-blue-900">Statistical Outliers</h4>
                    <p className="text-sm text-blue-700">Detect transactions beyond 3 standard deviations</p>
                  </div>
                  <div className="p-3 bg-green-50 rounded-lg">
                    <h4 className="font-medium text-green-900">Isolation Forest</h4>
                    <p className="text-sm text-green-700">Isolate anomalies using random forest technique</p>
                  </div>
                  <div className="p-3 bg-purple-50 rounded-lg">
                    <h4 className="font-medium text-purple-900">One-Class SVM</h4>
                    <p className="text-sm text-purple-700">Learn normal behavior patterns</p>
                  </div>
                  <div className="p-3 bg-orange-50 rounded-lg">
                    <h4 className="font-medium text-orange-900">Autoencoders</h4>
                    <p className="text-sm text-orange-700">Neural network-based anomaly detection</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Real-time Monitoring Tab */}
        {activeTab === 'monitoring' && (
          <div className="space-y-6">
            <div className="flex items-center gap-4 mb-6">
              <button
                onClick={() => setIsMonitoring(!isMonitoring)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  isMonitoring
                    ? 'bg-red-600 text-white hover:bg-red-700'
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
              </button>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                isMonitoring ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {isMonitoring ? 'ACTIVE' : 'INACTIVE'}
              </span>
            </div>

            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Live Transaction Stream</h3>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {realTimeData.length === 0 ? (
                  <p className="text-gray-500">No real-time data available. Start monitoring to see live transactions.</p>
                ) : (
                  realTimeData.map((transaction, index) => (
                    <div
                      key={index}
                      className={`p-3 rounded-lg border ${
                        transaction.isFraud
                          ? 'bg-red-50 border-red-200'
                          : 'bg-green-50 border-green-200'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <div>
                          <span className="font-medium">${transaction.amount}</span>
                          <span className="text-gray-600 ml-2">{transaction.merchantType}</span>
                          <span className="text-gray-500 ml-2">({transaction.timestamp})</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            transaction.isFraud
                              ? 'bg-red-100 text-red-800'
                              : 'bg-green-100 text-green-800'
                          }`}>
                            {transaction.isFraud ? 'FRAUD' : 'LEGITIMATE'}
                          </span>
                          <span className="text-sm text-gray-600">
                            Risk: {(transaction.riskScore * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Real-time Alerts Configuration</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">High Risk Threshold</label>
                  <input
                    type="range"
                    min="0.5"
                    max="1.0"
                    step="0.1"
                    defaultValue="0.8"
                    className="w-full"
                  />
                  <span className="text-sm text-gray-600">Transactions above 80% risk score trigger alerts</span>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Alert Frequency</label>
                  <select className="w-full px-3 py-2 border rounded-lg">
                    <option>Immediate</option>
                    <option>Every 5 minutes</option>
                    <option>Every 15 minutes</option>
                    <option>Hourly</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Feature Engineering Tab */}
        {activeTab === 'features' && (
          <div className="space-y-6">
            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white border rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-4">Feature Categories</h3>
                <div className="space-y-3">
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <h4 className="font-medium text-blue-900">Transaction Features</h4>
                    <p className="text-sm text-blue-700">Amount, time, merchant type, location</p>
                  </div>
                  <div className="p-3 bg-green-50 rounded-lg">
                    <h4 className="font-medium text-green-900">User Behavior Features</h4>
                    <p className="text-sm text-green-700">Historical patterns, frequency, preferences</p>
                  </div>
                  <div className="p-3 bg-purple-50 rounded-lg">
                    <h4 className="font-medium text-purple-900">Contextual Features</h4>
                    <p className="text-sm text-purple-700">Device info, IP address, geolocation</p>
                  </div>
                  <div className="p-3 bg-orange-50 rounded-lg">
                    <h4 className="font-medium text-orange-900">Derived Features</h4>
                    <p className="text-sm text-orange-700">Ratios, aggregations, time windows</p>
                  </div>
                </div>
              </div>

              <div style={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '8px', padding: '16px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px' }}>Kaggle Dataset Information</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div>
                    <h4 style={{ fontWeight: '500', marginBottom: '8px' }}>Dataset Overview</h4>
                    <p style={{ fontSize: '14px', color: '#6b7280' }}>
                      European cardholders transactions from September 2013 (284,807 transactions)
                    </p>
                  </div>
                  <div>
                    <h4 style={{ fontWeight: '500', marginBottom: '8px' }}>Feature Engineering</h4>
                    <p style={{ fontSize: '14px', color: '#6b7280' }}>
                      V1-V28: Principal Component Analysis (PCA) transformed features for privacy
                    </p>
                  </div>
                  <div>
                    <h4 style={{ fontWeight: '500', marginBottom: '8px' }}>Class Imbalance</h4>
                    <p style={{ fontSize: '14px', color: '#6b7280' }}>
                      Highly imbalanced: 99.828% normal, 0.172% fraud (492 fraudulent transactions)
                    </p>
                  </div>
                  <div>
                    <h4 style={{ fontWeight: '500', marginBottom: '8px' }}>Key Challenges</h4>
                    <p style={{ fontSize: '14px', color: '#6b7280' }}>
                      Extreme class imbalance, anonymized features, temporal patterns
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Feature Selection Methods</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-3 bg-gray-50 rounded-lg">
                  <h4 className="font-medium">Filter Methods</h4>
                  <p className="text-sm text-gray-600">Correlation, mutual information, chi-square</p>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <h4 className="font-medium">Wrapper Methods</h4>
                  <p className="text-sm text-gray-600">Forward selection, backward elimination</p>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <h4 className="font-medium">Embedded Methods</h4>
                  <p className="text-sm text-gray-600">LASSO, Ridge regression, tree-based</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FraudDetectionSystem;