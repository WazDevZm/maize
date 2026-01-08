import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Dimensions,
  RefreshControl,
  Alert,
  Image,
} from 'react-native';
import {
  Text,
  Card,
  Button,
  Searchbar,
  Chip,
  Surface,
  Menu,
  Divider,
  IconButton,
  FAB,
} from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import * as Animatable from 'react-native-animatable';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';

import { theme } from '../theme/theme';

const { width } = Dimensions.get('window');

interface HistoryItem {
  id: string;
  date: string;
  time: string;
  image: string;
  result: string;
  confidence: number;
  severity: 'None' | 'Low' | 'Medium' | 'High';
  treatment: string;
  location?: string;
}

const HistoryScreen: React.FC = () => {
  const navigation = useNavigation();
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [filterMenuVisible, setFilterMenuVisible] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState('All');
  const [sortMenuVisible, setSortMenuVisible] = useState(false);
  const [selectedSort, setSelectedSort] = useState('Newest First');

  const [historyData, setHistoryData] = useState<HistoryItem[]>([
    {
      id: '1',
      date: '2024-01-08',
      time: '14:30',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      result: 'Healthy',
      confidence: 98.5,
      severity: 'None',
      treatment: 'Continue current care practices',
      location: 'Field A - Section 1',
    },
    {
      id: '2',
      date: '2024-01-08',
      time: '11:15',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      result: 'Grey Leaf Spots',
      confidence: 94.2,
      severity: 'Medium',
      treatment: 'Apply fungicides, improve air circulation',
      location: 'Field B - Section 3',
    },
    {
      id: '3',
      date: '2024-01-07',
      time: '16:45',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      result: 'Leaf Blight',
      confidence: 91.8,
      severity: 'High',
      treatment: 'Apply copper-based fungicides, improve drainage',
      location: 'Field A - Section 2',
    },
    {
      id: '4',
      date: '2024-01-07',
      time: '09:20',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      result: 'Healthy',
      confidence: 96.8,
      severity: 'None',
      treatment: 'Continue current care practices',
      location: 'Field C - Section 1',
    },
    {
      id: '5',
      date: '2024-01-06',
      time: '13:10',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      result: 'MSV',
      confidence: 89.3,
      severity: 'High',
      treatment: 'Control leafhoppers, use resistant varieties',
      location: 'Field B - Section 1',
    },
    {
      id: '6',
      date: '2024-01-06',
      time: '10:30',
      image: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400&h=300&fit=crop',
      result: 'Healthy',
      confidence: 97.2,
      severity: 'None',
      treatment: 'Continue current care practices',
      location: 'Field A - Section 3',
    },
  ]);

  const [filteredData, setFilteredData] = useState<HistoryItem[]>(historyData);

  useEffect(() => {
    filterAndSortData();
  }, [searchQuery, selectedFilter, selectedSort, historyData]);

  const filterAndSortData = () => {
    let filtered = historyData;

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(item =>
        item.result.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.location?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply category filter
    if (selectedFilter !== 'All') {
      if (selectedFilter === 'Healthy') {
        filtered = filtered.filter(item => item.result === 'Healthy');
      } else if (selectedFilter === 'Diseased') {
        filtered = filtered.filter(item => item.result !== 'Healthy');
      } else if (selectedFilter === 'High Risk') {
        filtered = filtered.filter(item => item.severity === 'High');
      }
    }

    // Apply sorting
    filtered.sort((a, b) => {
      if (selectedSort === 'Newest First') {
        return new Date(b.date + ' ' + b.time).getTime() - new Date(a.date + ' ' + a.time).getTime();
      } else if (selectedSort === 'Oldest First') {
        return new Date(a.date + ' ' + a.time).getTime() - new Date(b.date + ' ' + b.time).getTime();
      } else if (selectedSort === 'Highest Confidence') {
        return b.confidence - a.confidence;
      } else if (selectedSort === 'Lowest Confidence') {
        return a.confidence - b.confidence;
      }
      return 0;
    });

    setFilteredData(filtered);
  };

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return '#dc3545';
      case 'medium': return '#ffc107';
      case 'low': return '#28a745';
      default: return '#28a745';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'alert-circle';
      case 'medium': return 'warning';
      case 'low': return 'checkmark-circle';
      default: return 'checkmark-circle';
    }
  };

  const handleItemPress = (item: HistoryItem) => {
    navigation.navigate('DetectionResult' as never, {
      image: item.image,
      result: {
        class: item.result,
        confidence: item.confidence,
        severity: item.severity,
        treatment: item.treatment,
      },
      fromHistory: true,
    } as never);
  };

  const handleDeleteItem = (id: string) => {
    Alert.alert(
      'Delete Detection',
      'Are you sure you want to delete this detection record?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            setHistoryData(prev => prev.filter(item => item.id !== id));
          },
        },
      ]
    );
  };

  const handleExportData = () => {
    Alert.alert(
      'Export Data',
      'Export detection history as CSV or PDF?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'CSV', onPress: () => console.log('Export CSV') },
        { text: 'PDF', onPress: () => console.log('Export PDF') },
      ]
    );
  };

  const getStatsData = () => {
    const total = historyData.length;
    const healthy = historyData.filter(item => item.result === 'Healthy').length;
    const diseased = total - healthy;
    const highRisk = historyData.filter(item => item.severity === 'High').length;
    
    return { total, healthy, diseased, highRisk };
  };

  const stats = getStatsData();

  return (
    <View style={styles.container}>
      {/* Header */}
      <LinearGradient
        colors={[theme.colors.primary, '#20c997']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>Detection History</Text>
          <Text style={styles.headerSubtitle}>
            {filteredData.length} of {historyData.length} records
          </Text>
        </View>
      </LinearGradient>

      {/* Stats Cards */}
      <Animatable.View animation="fadeInUp" delay={200} style={styles.statsContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <Surface style={[styles.statCard, { backgroundColor: theme.colors.primary }]} elevation={3}>
            <Text style={styles.statNumber}>{stats.total}</Text>
            <Text style={styles.statLabel}>Total Scans</Text>
          </Surface>
          
          <Surface style={[styles.statCard, { backgroundColor: theme.colors.success }]} elevation={3}>
            <Text style={styles.statNumber}>{stats.healthy}</Text>
            <Text style={styles.statLabel}>Healthy</Text>
          </Surface>
          
          <Surface style={[styles.statCard, { backgroundColor: theme.colors.warning }]} elevation={3}>
            <Text style={styles.statNumber}>{stats.diseased}</Text>
            <Text style={styles.statLabel}>Diseased</Text>
          </Surface>
          
          <Surface style={[styles.statCard, { backgroundColor: theme.colors.error }]} elevation={3}>
            <Text style={styles.statNumber}>{stats.highRisk}</Text>
            <Text style={styles.statLabel}>High Risk</Text>
          </Surface>
        </ScrollView>
      </Animatable.View>

      {/* Search and Filters */}
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search by disease or location..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          style={styles.searchBar}
          iconColor={theme.colors.primary}
        />
        
        <View style={styles.filterRow}>
          <Menu
            visible={filterMenuVisible}
            onDismiss={() => setFilterMenuVisible(false)}
            anchor={
              <Button
                mode="outlined"
                onPress={() => setFilterMenuVisible(true)}
                icon="filter"
                style={styles.filterButton}
              >
                {selectedFilter}
              </Button>
            }
          >
            <Menu.Item onPress={() => { setSelectedFilter('All'); setFilterMenuVisible(false); }} title="All" />
            <Menu.Item onPress={() => { setSelectedFilter('Healthy'); setFilterMenuVisible(false); }} title="Healthy" />
            <Menu.Item onPress={() => { setSelectedFilter('Diseased'); setFilterMenuVisible(false); }} title="Diseased" />
            <Menu.Item onPress={() => { setSelectedFilter('High Risk'); setFilterMenuVisible(false); }} title="High Risk" />
          </Menu>

          <Menu
            visible={sortMenuVisible}
            onDismiss={() => setSortMenuVisible(false)}
            anchor={
              <Button
                mode="outlined"
                onPress={() => setSortMenuVisible(true)}
                icon="sort"
                style={styles.filterButton}
              >
                Sort
              </Button>
            }
          >
            <Menu.Item onPress={() => { setSelectedSort('Newest First'); setSortMenuVisible(false); }} title="Newest First" />
            <Menu.Item onPress={() => { setSelectedSort('Oldest First'); setSortMenuVisible(false); }} title="Oldest First" />
            <Menu.Item onPress={() => { setSelectedSort('Highest Confidence'); setSortMenuVisible(false); }} title="Highest Confidence" />
            <Menu.Item onPress={() => { setSelectedSort('Lowest Confidence'); setSortMenuVisible(false); }} title="Lowest Confidence" />
          </Menu>
        </View>
      </View>

      {/* History List */}
      <ScrollView
        style={styles.listContainer}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        showsVerticalScrollIndicator={false}
      >
        {filteredData.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Ionicons name="document-text-outline" size={64} color="#ccc" />
            <Text style={styles.emptyText}>No detection records found</Text>
            <Text style={styles.emptySubtext}>
              {searchQuery ? 'Try adjusting your search or filters' : 'Start scanning leaves to build your history'}
            </Text>
          </View>
        ) : (
          filteredData.map((item, index) => (
            <Animatable.View
              key={item.id}
              animation="fadeInUp"
              delay={index * 100}
              style={styles.historyItem}
            >
              <Card style={styles.historyCard} onPress={() => handleItemPress(item)}>
                <Card.Content style={styles.cardContent}>
                  <View style={styles.cardHeader}>
                    <View style={styles.imageContainer}>
                      <Image source={{ uri: item.image }} style={styles.leafImage} />
                    </View>
                    
                    <View style={styles.detectionInfo}>
                      <View style={styles.resultHeader}>
                        <Text style={styles.resultText}>{item.result}</Text>
                        <Chip
                          style={[
                            styles.severityChip,
                            { backgroundColor: getSeverityColor(item.severity) }
                          ]}
                          textStyle={styles.severityText}
                        >
                          <Ionicons
                            name={getSeverityIcon(item.severity) as any}
                            size={12}
                            color="white"
                          />
                          {' '}{item.severity}
                        </Chip>
                      </View>
                      
                      <Text style={styles.confidenceText}>
                        Confidence: {item.confidence.toFixed(1)}%
                      </Text>
                      
                      <Text style={styles.dateText}>
                        {item.date} at {item.time}
                      </Text>
                      
                      {item.location && (
                        <Text style={styles.locationText}>
                          📍 {item.location}
                        </Text>
                      )}
                    </View>
                    
                    <IconButton
                      icon="delete"
                      iconColor="#dc3545"
                      size={20}
                      onPress={() => handleDeleteItem(item.id)}
                    />
                  </View>
                  
                  <Divider style={styles.divider} />
                  
                  <Text style={styles.treatmentText} numberOfLines={2}>
                    💊 {item.treatment}
                  </Text>
                </Card.Content>
              </Card>
            </Animatable.View>
          ))
        )}
        
        <View style={styles.bottomSpacing} />
      </ScrollView>

      {/* Floating Action Button */}
      <FAB
        icon="camera"
        style={styles.fab}
        onPress={() => navigation.navigate('Camera' as never)}
        label="New Scan"
      />

      {/* Export Button */}
      {historyData.length > 0 && (
        <View style={styles.exportContainer}>
          <Button
            mode="outlined"
            onPress={handleExportData}
            icon="download"
            style={styles.exportButton}
          >
            Export Data
          </Button>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  headerContent: {
    alignItems: 'center',
  },
  headerTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  headerSubtitle: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    marginTop: 4,
  },
  statsContainer: {
    paddingVertical: 15,
    paddingLeft: 20,
  },
  statCard: {
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 12,
    marginRight: 15,
    minWidth: 100,
    alignItems: 'center',
  },
  statNumber: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  statLabel: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 12,
    marginTop: 4,
  },
  searchContainer: {
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  searchBar: {
    marginBottom: 10,
    backgroundColor: 'white',
  },
  filterRow: {
    flexDirection: 'row',
    gap: 10,
  },
  filterButton: {
    flex: 1,
  },
  listContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 18,
    color: '#666',
    marginTop: 16,
    fontWeight: '600',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 8,
    textAlign: 'center',
    paddingHorizontal: 40,
  },
  historyItem: {
    marginBottom: 12,
  },
  historyCard: {
    borderRadius: 12,
    backgroundColor: 'white',
  },
  cardContent: {
    padding: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  imageContainer: {
    marginRight: 15,
  },
  leafImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  detectionInfo: {
    flex: 1,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  resultText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  severityChip: {
    borderRadius: 12,
    height: 24,
  },
  severityText: {
    color: 'white',
    fontSize: 10,
    fontWeight: '600',
  },
  confidenceText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  dateText: {
    fontSize: 12,
    color: '#999',
    marginBottom: 2,
  },
  locationText: {
    fontSize: 12,
    color: theme.colors.primary,
  },
  divider: {
    marginVertical: 12,
  },
  treatmentText: {
    fontSize: 13,
    color: '#666',
    lineHeight: 18,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 80,
    backgroundColor: theme.colors.primary,
  },
  exportContainer: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 80,
  },
  exportButton: {
    borderColor: theme.colors.primary,
  },
  bottomSpacing: {
    height: 100,
  },
});

export default HistoryScreen;