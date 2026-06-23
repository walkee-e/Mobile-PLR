import React, {useEffect, useState} from 'react';
import {View, StyleSheet, ScrollView, Text, Dimensions} from 'react-native';
import {LineChart} from 'react-native-chart-kit';

import {cacheSession} from '../api';

const HEX_NAME = {
  '#FF0000': 'Red',
  '#00FF00': 'Green',
  '#0000FF': 'Blue',
  '#FFFF00': 'Yellow',
  '#FFFFFF': 'White',
};

function chartPoints(series, maxPoints = 30) {
  if (!series || series.length === 0) return [];
  const step = Math.max(1, Math.ceil(series.length / maxPoints));
  return series.filter((_, index) => index % step === 0);
}

export function ResultsScreen({route}) {
  const {experimentData} = route.params;
  const summary = experimentData.summary || {};
  const rawResults = summary.results || summary;
  const [cachedPath, setCachedPath] = useState(null);

  const entries = Object.entries(rawResults)
    .filter(([, result]) => result && result._meta)
    .map(([clipPath, result]) => ({
      clipPath,
      status: result.status,
      error: result.error,
      hex: result?._meta?.hex_color ?? '#000000',
      led: result?._meta?.led_index ?? 0,
      eye: result?._meta?.eye ?? '',
      baseline: result?.baseline_diameter_px,
      minimum: result?.min_diameter_px,
      amplitude: result?.constriction_amplitude_px,
      latency: result?.latency_ms,
      series: result?.diameter_series || [],
      formula: result?.formula_version,
    }));

  useEffect(() => {
    if (entries.length === 0) return;
    cacheSession(
      {
        name: experimentData.name,
        age: experimentData.age,
        sex: experimentData.sex,
        controlMode: experimentData.controlMode,
        modeLabel: experimentData.modeLabel,
        schedule: experimentData.schedule,
        time: new Date().toISOString(),
      },
      summary,
    )
      .then(setCachedPath)
      .catch(e => console.warn('cacheSession failed', e));
  }, []);

  const successful = entries.filter(entry => entry.status === 'ok');
  const chartLabels = successful.map((entry, index) =>
    index % 2 === 0 ? HEX_NAME[entry.hex] || entry.hex.slice(1, 4) : '',
  );

  return (
    <ScrollView style={{backgroundColor: '#F7F7F7'}}>
      <Text style={styles.h1}>Session</Text>
      <Text style={styles.status}>
        Result status: {summary.status || 'legacy'} · Unit: pixels
      </Text>

      <Text style={styles.line}>Name: {experimentData.name}</Text>
      <Text style={styles.line}>Age: {experimentData.age}</Text>
      <Text style={styles.line}>
        Control mode: {experimentData.modeLabel || experimentData.controlMode}
      </Text>
      {experimentData.controlMode === 'dual' ? (
        <Text style={styles.line}>
          Flashes: {experimentData.schedule?.flashes?.length || 0} · Break:{' '}
          {experimentData.schedule?.gap}s
        </Text>
      ) : (
        <>
          <Text style={styles.line}>
            Rounds: {experimentData.schedule?.rounds} · Color:{' '}
            {experimentData.schedule?.color}
          </Text>
          <Text style={styles.line}>
            Duration: {experimentData.schedule?.duration}s · Eye pause:{' '}
            {experimentData.schedule?.innerPause}s · Round break:{' '}
            {experimentData.schedule?.gap}s
          </Text>
        </>
      )}

      <Text style={styles.note}>
        Initial formulas: median smoothed pre-flash baseline; minimum smoothed
        post-flash value; amplitude = baseline − minimum; latency = flash onset
        to minimum. These formulas still require clinical validation.
      </Text>

      <Text style={styles.h1}>Per-flash results</Text>
      {entries.length === 0 && <Text style={styles.line}>No results returned.</Text>}

      {entries.map((entry, index) => {
        const points = chartPoints(entry.series);
        return (
          <View key={`${entry.clipPath}-${index}`} style={styles.card}>
            <Text style={styles.cardTitle}>
              {entry.eye || `LED ${entry.led}`} · {HEX_NAME[entry.hex] || entry.hex}
            </Text>
            {entry.status !== 'ok' ? (
              <Text style={styles.error}>Inference failed: {entry.error}</Text>
            ) : (
              <>
                <Text style={styles.cardLine}>
                  Baseline: {Number(entry.baseline).toFixed(2)} px · Minimum:{' '}
                  {Number(entry.minimum).toFixed(2)} px
                </Text>
                <Text style={styles.cardLine}>
                  Amplitude: {Number(entry.amplitude).toFixed(2)} px · Latency:{' '}
                  {Number(entry.latency).toFixed(0)} ms
                </Text>
                {points.length > 1 && (
                  <LineChart
                    data={{
                      labels: points.map((point, pointIndex) =>
                        pointIndex % 5 === 0
                          ? `${(point.time_ms / 1000).toFixed(1)}s`
                          : '',
                      ),
                      datasets: [
                        {data: points.map(point => Number(point.smoothed_diameter_px))},
                      ],
                    }}
                    width={Dimensions.get('window').width - 72}
                    height={190}
                    yAxisSuffix=" px"
                    chartConfig={chartConfig}
                    bezier
                    style={styles.chart}
                  />
                )}
              </>
            )}
          </View>
        );
      })}

      {successful.length > 1 && (
        <View style={styles.summaryChart}>
          <Text style={styles.cardTitle}>Per-flash comparison</Text>
          <LineChart
            data={{
              labels: chartLabels,
              datasets: [
                {
                  data: successful.map(entry => Number(entry.baseline) || 0),
                  color: () => 'rgba(40,90,200,1)',
                },
                {
                  data: successful.map(entry => Number(entry.minimum) || 0),
                  color: () => 'rgba(200,60,60,1)',
                },
              ],
              legend: ['baseline px', 'minimum px'],
            }}
            width={Dimensions.get('window').width - 40}
            height={220}
            yAxisSuffix=" px"
            chartConfig={chartConfig}
            bezier
            style={styles.chart}
          />
        </View>
      )}

      {cachedPath && <Text style={styles.cached}>Saved to {cachedPath}</Text>}
    </ScrollView>
  );
}

const chartConfig = {
  backgroundGradientFrom: '#ffffff',
  backgroundGradientTo: '#ffffff',
  decimalPlaces: 1,
  color: (opacity = 1) => `rgba(0,0,0,${opacity})`,
  labelColor: (opacity = 1) => `rgba(0,0,0,${opacity})`,
  propsForDots: {r: '2'},
};

const styles = StyleSheet.create({
  h1: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'black',
    textAlign: 'center',
    marginTop: 24,
  },
  status: {fontSize: 14, color: '#333', textAlign: 'center', marginTop: 8},
  line: {fontSize: 15, color: 'black', textAlign: 'center', marginTop: 10},
  note: {
    fontSize: 12,
    color: '#555',
    marginHorizontal: 24,
    marginTop: 18,
    lineHeight: 18,
  },
  card: {
    marginHorizontal: 20,
    marginTop: 12,
    padding: 12,
    backgroundColor: '#ffffff',
    borderRadius: 10,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 4,
  },
  cardTitle: {fontSize: 16, fontWeight: 'bold', color: 'black', marginBottom: 6},
  cardLine: {fontSize: 14, color: '#333', marginTop: 2},
  error: {fontSize: 14, color: '#a22', marginTop: 4},
  chart: {borderRadius: 16, marginTop: 12},
  summaryChart: {marginHorizontal: 20, marginVertical: 20},
  cached: {fontSize: 12, color: '#666', textAlign: 'center', marginVertical: 16},
});
