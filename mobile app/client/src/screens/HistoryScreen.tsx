import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  StyleSheet,
  FlatList,
  Image,
  Alert,
  RefreshControl,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Text,
  Button,
  Surface,
  Chip,
  IconButton,
  Searchbar,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';

import { theme } from '@/theme/theme';
import type { RootTabParamList, DetectionHistory } from '@/types';
import { getDetectionHistory, clearDetectionHistory, deleteDetectionFromHistory } from '@/utils/storage';

type Props = BottomTabScreenProps<RootTabParamList, 'History'>;

const HistoryScreen: React.FC<Props> = ({ navigation }) => {
  const [history, setHistory] = useState<DetectionHistory[]>([]);
  const [filteredHistory, setFilteredHistory] = useState<DetectionHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    filterHistory();
  }, [history, searchQuery]);

  const loadHistory = async (): Promise<void> => {
    try {
      setLoading(true);
      const historyData = await getDetectionHistory();
      setHistory(historyData);
    } catch (error) {
      console.error('Error loading history:', error);
      Alert.alert('Error', 'Failed to load detection history');
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadHistory();
    setRefreshing(false);
  }, []);

  const filterHistory = (): void => {
    if (!searchQuery.trim()) {
      setFilteredHistory(history);
      return;
    }

    const filtered = history.filter(item => {
      const query = searchQuery.toLowerCase();
      const healthStatus = item.result.health_status.toLowerCase();
      const predictions = item.result.predictions.map(p => p.class.toLowerCase()).join(' ');
      
      return healthStatus.includes(query) || predictions.includes(query);
    });

    setFilteredHistory(filtered);
  };

  const handleDeleteItem = (id: string): void => {
    Alert.alert(
      'Delete Detection',
      'Are you sure you want to delete this detection from history?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await deleteDetectionFromHistory(id);
              await loadHistory();
            } catch (error) {
              Alert.alert('Error', 'Failed to delete detection');
            }
          },
        },
      ]
    );
  };

  const handleClearAll = (): void => {
    Alert.alert(
      'Clear All History',
      'Are you sure you want to delete all detection history? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: async () => {
            try {
              await clearDetectionHistory();
              await loadHistory();
            } catch (error) {
              Alert.alert('Error', 'Failed to clear history');
            }
          },
        },
      ]
    );
  };

  const formatDate = (timestamp: string): string => {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getHealthStatusColor = (healthStatus: string): string => {
    if (healthStatus.toLowerCase().includes('healthy')) {
      return theme.colors.success;
    }
    if (healthStatus.toLowerCase().includes('disease')) {
      return theme.colors.error;
    }
    return theme.colors.warning;
  };

  const renderHistoryItem = ({ item, index }: { item: DetectionHistory; index: number }): JSX.Element => {
    const hasDisease = item.result.predictions.some(p => p.class !== 'Health');
    const mainPrediction = item.result.predictions.length > 0 
      ? item.result.predictions.reduce((prev, current) => 
          prev.confidence > current.confidence ? prev : current
        )
      : null;

    return (
      <Animatable.View
        animation="fadeInUp"
        delay={index * 100}
        duration={600}
      >
        <Card style={styles.historyCard}>
          <View style={styles.cardContent}>
            <Image source={{ uri: item.imageUri }} style={styles.historyImage} />
            
            <View style={styles.historyInfo}>
              <View style={styles.historyHeader}>
                <Text style={styles.historyDate}>{formatDate(item.timestamp)}</Text>
                <IconButton
                  icon="delete"
                  size={20}
                  onPress={() => handleDeleteItem(item.id)}
                  style={styles.deleteButton}
                />
              </View>
              
              <View style={styles.statusContainer}>
                <Ionicons
                  name={hasDisease ? "warning" : "checkmark-circle"}
                  size={20}
                  color={getHealthStatusColor(item.result.health_status)}
                />
                <Text style={[
                  styles.healthStatus,
                  { color: getHealthStatusColor(item.result.health_status) }
                ]}>
                  {item.result.health_status}
                </Text>
              </View>

              {mainPrediction && (
                <View style={styles.predictionContainer}>
                  <Chip
                    style={[
                      styles.predictionChip,
                      { backgroundColor: mainPrediction.disease_info?.color + '20' || theme.colors.primary + '20' }
                    ]}
                    textStyle={styles.chipText}
                  >
                    {mainPrediction.class}
                  </Chip>
                  <Text style={styles.confidenceText}>
                    {(mainPrediction.confidence * 100).toFixed(1)}%
                  </Text>
                </View>
              )}

              <Text style={styles.severityText}>
                Severity: {item.result.severity} | Detections: {item.result.total_detections}
              </Text>
            </View>
          </View>
        </Card>
      </Animatable.View>
    );
  };

  const renderEmptyState = (): JSX.Element => (
    <View style={styles.emptyContainer}>
      <Ionicons name="time-outline" size={64} color={theme.colors.primary} />
      <Title style={styles.emptyTitle}>No Detection History</Title>
      <Paragraph style={styles.emptyText}>
        Start detecting diseases in maize leaves to see your history here.
      </Paragraph>
      <Button
        mode="contained"
        onPress={() => navigation.navigate('Camera')}
        style={styles.emptyButton}
      >
        Start Detection
      </Button>
    </View>
  );

  const renderHeader = (): JSX.Element => (
    <View style={styles.header}>
      <Surface style={styles.statsCard} elevation={2}>
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{history.length}</Text>
            <Text style={styles.statLabel}>Total Scans</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>
              {history.filter(h => h.result.predictions.some(p => p.class !== 'Health')).length}
            </Text>
            <Text style={styles.statLabel}>Diseases Found</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>
              {history.filter(h => h.result.health_status.toLowerCase().includes('healthy')).length}
            </Text>
            <Text style={styles.statLabel}>Healthy Scans</Text>
          </View>
        </View>
      </Surface>

      {history.length > 0 && (
        <>
          <Searchbar
            placeholder="Search detections..."
            onChangeText={setSearchQuery}
            value={searchQuery}
            style={styles.searchBar}
          />
          
          <View style={styles.actionRow}>
            <Text style={styles.resultCount}>
              {filteredHistory.length} of {history.length} detections
            </Text>
            <Button
              mode="outlined"
              onPress={handleClearAll}
              style={styles.clearButton}
              compact
            >
              Clear All
            </Button>
          </View>
        </>
      )}
    </View>
  );

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="time" size={64} color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading history...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={filteredHistory}
        renderItem={renderHistoryItem}
        keyExtractor={(item) => item.id}
        ListHeaderComponent={renderHeader}
        ListEmptyComponent={history.length === 0 ? renderEmptyState : (
          <View style={styles.noResultsContainer}>
            <Ionicons name="search-outline" size={48} color={theme.colors.primary} />
            <Text style={styles.noResultsText}>No matching detections found</Text>
          </View>
        )}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            colors={[theme.colors.primary]}
          />
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    fontSize: 16,
    marginTop: 16,
    color: theme.colors.onBackground,
  },
  listContent: {
    padding: 16,
  },
  header: {
    marginBottom: 16,
  },
  statsCard: {
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.colors.primary,
  },
  statLabel: {
    fontSize: 12,
    color: theme.colors.onSurface,
    opacity: 0.7,
    marginTop: 4,
  },
  statDivider: {
    width: 1,
    height: 40,
    backgroundColor: theme.colors.outline,
    opacity: 0.3,
  },
  searchBar: {
    marginBottom: 16,
    borderRadius: 25,
  },
  actionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  resultCount: {
    fontSize: 14,
    color: theme.colors.onBackground,
    opacity: 0.7,
  },
  clearButton: {
    borderRadius: 20,
  },
  historyCard: {
    marginBottom: 12,
    borderRadius: 12,
    overflow: 'hidden',
  },
  cardContent: {
    flexDirection: 'row',
    padding: 12,
  },
  historyImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
    marginRight: 12,
  },
  historyInfo: {
    flex: 1,
    justifyContent: 'space-between',
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  historyDate: {
    fontSize: 12,
    color: theme.colors.onSurface,
    opacity: 0.7,
  },
  deleteButton: {
    margin: 0,
    padding: 4,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },
  healthStatus: {
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 6,
  },
  predictionContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginVertical: 4,
  },
  predictionChip: {
    alignSelf: 'flex-start',
  },
  chipText: {
    fontSize: 12,
  },
  confidenceText: {
    fontSize: 12,
    fontWeight: '600',
    color: theme.colors.primary,
  },
  severityText: {
    fontSize: 11,
    color: theme.colors.onSurface,
    opacity: 0.6,
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyTitle: {
    textAlign: 'center',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyText: {
    textAlign: 'center',
    marginBottom: 24,
    opacity: 0.7,
  },
  emptyButton: {
    borderRadius: 25,
  },
  noResultsContainer: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  noResultsText: {
    fontSize: 16,
    marginTop: 12,
    color: theme.colors.onBackground,
    opacity: 0.7,
  },
});

export default HistoryScreen;