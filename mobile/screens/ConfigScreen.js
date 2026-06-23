import React, {useState} from 'react';
import {
  View,
  StyleSheet,
  Text,
  Button,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import Slider from '@react-native-community/slider';

import {startSession} from '../api';

const MODES = [
  {id: 'dual', label: 'Dual'},
  {id: 'left_to_right', label: 'Left → Right'},
  {id: 'right_to_left', label: 'Right → Left'},
];

const COLORS = [
  {name: 'Red', hex: '#FF0000', swatch: '#EB5353'},
  {name: 'Green', hex: '#00FF00', swatch: '#36AE7C'},
  {name: 'Blue', hex: '#0000FF', swatch: '#187498'},
  {name: 'Yellow', hex: '#FFFF00', swatch: '#F9D923'},
  {name: 'Magenta', hex: '#FF00FF', swatch: '#D946EF'},
  {name: 'Cyan', hex: '#00FFFF', swatch: '#22D3EE'},
  {name: 'White', hex: '#FFFFFF', swatch: '#F7F7F7'},
];

const DEFAULT_FLASHES = [
  {color: 'Red', hex: '#FF0000', duration: 1},
  {color: 'Green', hex: '#00FF00', duration: 1},
  {color: 'Blue', hex: '#0000FF', duration: 1},
];

function ChoiceButton({selected, label, onPress}) {
  return (
    <TouchableOpacity
      style={[styles.choice, selected && styles.choiceSelected]}
      onPress={onPress}>
      <Text style={[styles.choiceText, selected && styles.choiceTextSelected]}>
        {label}
      </Text>
    </TouchableOpacity>
  );
}

function ColorSelector({value, onChange}) {
  return (
    <View style={styles.colorRow}>
      {COLORS.map(color => (
        <TouchableOpacity
          key={color.name}
          accessibilityLabel={color.name}
          style={[
            styles.colorCircle,
            {backgroundColor: color.swatch},
            value === color.name && styles.colorSelected,
          ]}
          onPress={() => onChange(color)}
        />
      ))}
    </View>
  );
}

function ValueSlider({
  label,
  value,
  minimum,
  maximum,
  step = 1,
  suffix = '',
  onChange,
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>
        {label}: {value}
        {suffix}
      </Text>
      <Slider
        step={step}
        minimumValue={minimum}
        maximumValue={maximum}
        value={value}
        onValueChange={onChange}
        minimumTrackTintColor="#8192A6"
        maximumTrackTintColor="#d3d3d3"
        thumbTintColor="#474747"
      />
    </View>
  );
}

export function ConfigScreen({navigation, route}) {
  const {data} = route.params;
  const [mode, setMode] = useState('dual');
  const [dualFlashes, setDualFlashes] = useState(DEFAULT_FLASHES);
  const [sequentialColor, setSequentialColor] = useState(COLORS[0]);
  const [rounds, setRounds] = useState(3);
  const [duration, setDuration] = useState(1);
  const [innerPause, setInnerPause] = useState(1);
  const [gap, setGap] = useState(3);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  function updateFlash(index, changes) {
    setDualFlashes(current =>
      current.map((flash, flashIndex) =>
        flashIndex === index ? {...flash, ...changes} : flash,
      ),
    );
  }

  function addFlash() {
    setDualFlashes(current => [
      ...current,
      {color: 'White', hex: '#FFFFFF', duration: 1},
    ]);
  }

  function removeFlash(index) {
    if (dualFlashes.length <= 1) return;
    setDualFlashes(current =>
      current.filter((_, flashIndex) => flashIndex !== index),
    );
  }

  async function startExperiment() {
    setSubmitting(true);
    setError('');

    const schedule =
      mode === 'dual'
        ? {
            flashes: dualFlashes.map(flash => ({
              hex: flash.hex,
              duration: flash.duration,
            })),
            gap,
          }
        : {
            rounds,
            hex: sequentialColor.hex,
            color: sequentialColor.name,
            duration,
            innerPause,
            gap,
          };

    const payload = {
      participant: {name: data.Name, age: data.Age, sex: data.Sex},
      controlMode: mode,
      schedule,
    };

    const experimentData = {
      name: data.Name,
      age: data.Age,
      sex: data.Sex,
      time: '',
      controlMode: mode,
      modeLabel: MODES.find(item => item.id === mode)?.label,
      schedule,
      pupilDiameters: '',
    };

    try {
      await startSession(payload);
      navigation.navigate('Experiment', {experimentData});
    } catch (requestError) {
      console.warn('startSession failed', requestError);
      setError('Could not start the session. Check the Pi connection.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <ScrollView
      style={styles.page}
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled">
      <Text style={styles.heading}>Control mode</Text>
      <View style={styles.modeRow}>
        {MODES.map(item => (
          <ChoiceButton
            key={item.id}
            label={item.label}
            selected={mode === item.id}
            onPress={() => setMode(item.id)}
          />
        ))}
      </View>

      {mode === 'dual' ? (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Dual flash schedule</Text>
          <Text style={styles.help}>
            Both LEDs use the same color at the same time. Three flashes are
            provided by default.
          </Text>

          {dualFlashes.map((flash, index) => (
            <View key={`flash-${index}`} style={styles.card}>
              <View style={styles.cardHeader}>
                <Text style={styles.cardTitle}>Flash {index + 1}</Text>
                {dualFlashes.length > 1 && (
                  <TouchableOpacity onPress={() => removeFlash(index)}>
                    <Text style={styles.remove}>Remove</Text>
                  </TouchableOpacity>
                )}
              </View>
              <Text style={styles.fieldLabel}>
                Color: {flash.color} ({flash.hex})
              </Text>
              <ColorSelector
                value={flash.color}
                onChange={color =>
                  updateFlash(index, {color: color.name, hex: color.hex})
                }
              />
              <ValueSlider
                label="Duration"
                value={flash.duration}
                minimum={0.1}
                maximum={10}
                step={0.1}
                suffix="s"
                onChange={value => updateFlash(index, {duration: value})}
              />
            </View>
          ))}

          <View style={styles.addButton}>
            <Button title="Add another flash" color="#8192A6" onPress={addFlash} />
          </View>
          <ValueSlider
            label="Break between flashes"
            value={gap}
            minimum={3}
            maximum={15}
            suffix="s"
            onChange={setGap}
          />
        </View>
      ) : (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            {mode === 'left_to_right' ? 'Left → Right' : 'Right → Left'} schedule
          </Text>
          <Text style={styles.help}>
            {mode === 'left_to_right'
              ? 'Left LED flashes first, followed by the pause, then the right LED.'
              : 'Right LED flashes first, followed by the pause, then the left LED.'}
          </Text>
          <Text style={styles.fieldLabel}>
            Color: {sequentialColor.name} ({sequentialColor.hex})
          </Text>
          <ColorSelector value={sequentialColor.name} onChange={setSequentialColor} />
          <ValueSlider
            label="Rounds"
            value={rounds}
            minimum={1}
            maximum={10}
            onChange={setRounds}
          />
          <ValueSlider
            label="Flash duration"
            value={duration}
            minimum={0.1}
            maximum={10}
            step={0.1}
            suffix="s"
            onChange={setDuration}
          />
          <ValueSlider
            label="Pause between eyes"
            value={innerPause}
            minimum={0.1}
            maximum={10}
            step={0.1}
            suffix="s"
            onChange={setInnerPause}
          />
          <ValueSlider
            label="Break between rounds"
            value={gap}
            minimum={3}
            maximum={15}
            suffix="s"
            onChange={setGap}
          />
        </View>
      )}

      {error ? <Text style={styles.error}>{error}</Text> : null}
      <View style={styles.startButton}>
        <Button
          color="#81A695"
          title={submitting ? 'Starting…' : 'Start Experiment'}
          disabled={submitting}
          onPress={startExperiment}
        />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: {backgroundColor: '#F7F7F7'},
  container: {padding: 20, paddingBottom: 48},
  heading: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#111',
    textAlign: 'center',
    marginBottom: 14,
  },
  modeRow: {flexDirection: 'row', justifyContent: 'center', flexWrap: 'wrap'},
  choice: {
    borderWidth: 1,
    borderColor: '#8192A6',
    borderRadius: 18,
    paddingVertical: 9,
    paddingHorizontal: 13,
    margin: 4,
  },
  choiceSelected: {backgroundColor: '#8192A6'},
  choiceText: {color: '#334', fontWeight: '600'},
  choiceTextSelected: {color: '#fff'},
  section: {marginTop: 22},
  sectionTitle: {fontSize: 20, fontWeight: 'bold', color: '#111'},
  help: {fontSize: 13, color: '#555', lineHeight: 19, marginTop: 6, marginBottom: 12},
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
  },
  cardHeader: {flexDirection: 'row', justifyContent: 'space-between'},
  cardTitle: {fontSize: 17, fontWeight: 'bold', color: '#222'},
  remove: {color: '#a33', fontWeight: '600'},
  field: {marginTop: 12},
  fieldLabel: {fontSize: 15, fontWeight: '600', color: '#222', marginTop: 8},
  colorRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 9,
    marginBottom: 3,
  },
  colorCircle: {
    width: 34,
    height: 34,
    borderRadius: 17,
    borderWidth: 1,
    borderColor: '#888',
    marginRight: 12,
    marginBottom: 8,
  },
  colorSelected: {borderWidth: 4, borderColor: '#8192A6'},
  addButton: {marginVertical: 8},
  startButton: {marginTop: 28},
  error: {color: '#a22', textAlign: 'center', marginTop: 16},
});
