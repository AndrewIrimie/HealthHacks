import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'socket_audio_service.dart';

void main() => runApp(const DocDescribeApp());

class DocDescribeApp extends StatelessWidget {
  const DocDescribeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Doc Describe',
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.indigo,
        brightness: Brightness.dark,
      ),
      home: const RecordPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class RecordPage extends StatefulWidget {
  const RecordPage({super.key});
  @override
  State<RecordPage> createState() => _RecordPageState();
}

class _RecordPageState extends State<RecordPage>
    with SingleTickerProviderStateMixin {
  
  // Audio recording
  final AudioRecorder _recorder = AudioRecorder();
  bool _isRecording = false;
  StreamSubscription<Uint8List>? _audioStreamSubscription;
  
  // Socket communication
  final SocketAudioService _socketService = SocketAudioService();
  bool _isConnected = false;
  
  // UI state
  String _status = ConnectionStatus.idle;
  Duration _elapsed = Duration.zero;
  Timer? _timer;
  late final AnimationController _pulse;
  
  // Connection settings
  String _serverHost = 'localhost'; // Change to your computer's IP for device testing
  int _serverPort = 8080;
  
  // Statistics
  int _bytesSent = 0;
  int _packetsSpent = 0;

  @override
  void initState() {
    super.initState();
    _setupAnimation();
    _setupSocketService();
    _loadConnectionSettings();
  }

  void _setupAnimation() {
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
      lowerBound: 0.9,
      upperBound: 1.08,
    )..addStatusListener((s) {
      if (s == AnimationStatus.completed) _pulse.reverse();
      if (s == AnimationStatus.dismissed) _pulse.forward();
    });
  }

  void _setupSocketService() {
    _socketService.onStatusUpdate = (status) {
      if (mounted) setState(() => _status = status);
    };
    
    _socketService.onError = (error) {
      if (mounted) {
        setState(() => _status = 'Error: $error');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Socket Error: $error'), backgroundColor: Colors.red),
        );
      }
    };
    
    _socketService.onConnected = () {
      if (mounted) {
        setState(() {
          _isConnected = true;
          _status = ConnectionStatus.connected;
        });
        _socketService.startHeartbeat();
      }
    };
    
    _socketService.onDisconnected = () {
      if (mounted) {
        setState(() {
          _isConnected = false;
          _status = ConnectionStatus.idle;
        });
        _socketService.stopHeartbeat();
      }
    };
  }

  Future<void> _loadConnectionSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _serverHost = prefs.getString('server_host') ?? 'localhost';
      _serverPort = prefs.getInt('server_port') ?? 8080;
    });
    _socketService.updateServerSettings(_serverHost, _serverPort);
  }

  Future<void> _saveConnectionSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('server_host', _serverHost);
    await prefs.setInt('server_port', _serverPort);
    _socketService.updateServerSettings(_serverHost, _serverPort);
  }

  @override
  void dispose() {
    _timer?.cancel();
    _pulse.dispose();
    _audioStreamSubscription?.cancel();
    _recorder.dispose();
    _socketService.dispose();
    super.dispose();
  }

  Future<void> _toggleRecording() async =>
      _isRecording ? _stopRecording() : _startRecording();

  Future<void> _startRecording() async {
    // Check microphone permission
    try {
      final hasPerm = await _recorder.hasPermission();
      if (!hasPerm) {
        setState(() => _status = 'Microphone permission denied');
        return;
      }
    } catch (_) {
      setState(() => _status = 'Permission check failed');
      return;
    }

    // Connect to server if not connected
    if (!_isConnected) {
      final connected = await _socketService.connect();
      if (!connected) {
        return; // Error handling is done in socket service callbacks
      }
      // Wait a moment for connection to stabilize
      await Future.delayed(const Duration(milliseconds: 500));
    }

    try {
      // Start session on server
      final sessionStarted = _socketService.sendStartSession(metadata: {
        'audio_format': 'pcm16',
        'sample_rate': 16000,
        'channels': 1,
        'client_info': 'Flutter DocDescribe App',
      });

      if (!sessionStarted) {
        setState(() => _status = 'Failed to start session');
        return;
      }

      // Configure audio recording for real-time streaming
      const config = RecordConfig(
        encoder: AudioEncoder.pcm16bits, // Raw PCM data
        sampleRate: 16000,               // Match server expectation
        numChannels: 1,                  // Mono audio
        autoGain: true,                  // Automatic gain control
        echoCancel: true,                // Echo cancellation
        noiseSuppress: true,             // Noise suppression
      );

      // Start real-time audio stream
      final audioStream = await _recorder.startStream(config);
      
      // Reset statistics
      _bytesSent = 0;
      _packetsSpent = 0;

      // Listen to audio stream and send to server
      _audioStreamSubscription = audioStream.listen(
        (audioData) {
          if (_isConnected && _socketService.sendAudioData(audioData)) {
            _bytesSent += audioData.length;
            _packetsSpent++;
          }
        },
        onError: (error) {
          setState(() => _status = 'Audio stream error: $error');
        },
      );

      setState(() {
        _isRecording = true;
        _status = ConnectionStatus.recording;
        _elapsed = Duration.zero;
      });

      // Start timer for elapsed time
      _timer?.cancel();
      _timer = Timer.periodic(
        const Duration(seconds: 1),
        (_) => setState(() => _elapsed += const Duration(seconds: 1)),
      );
      
      _pulse.forward();

    } catch (e) {
      setState(() => _status = 'Failed to start recording: $e');
    }
  }

  Future<void> _stopRecording() async {
    try {
      // Stop audio recording
      await _recorder.stop();
      await _audioStreamSubscription?.cancel();
      _audioStreamSubscription = null;
      
      // Stop timer and animation
      _timer?.cancel();
      _pulse.stop();

      // Send end session to server
      _socketService.sendEndSession();

      setState(() {
        _isRecording = false;
        _status = 'Stopped - Sent ${_formatBytes(_bytesSent)} in $_packetsSpent packets';
      });

      // Show completion message
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Recording sent: ${_formatBytes(_bytesSent)} in ${_fmt(_elapsed)}'),
            backgroundColor: Colors.green,
          ),
        );
      }

    } catch (e) {
      setState(() => _status = 'Failed to stop: $e');
    }
  }

  Future<void> _toggleConnection() async {
    if (_isConnected) {
      await _socketService.disconnect();
    } else {
      await _socketService.connect();
    }
  }

  String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final isRec = _isRecording;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Doc Describe'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(_isConnected ? Icons.wifi : Icons.wifi_off),
            onPressed: _toggleConnection,
            tooltip: _isConnected ? 'Disconnect' : 'Connect',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Connection Settings
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Server Connection',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          flex: 3,
                          child: TextField(
                            controller: TextEditingController(text: _serverHost),
                            onChanged: (v) => _serverHost = v,
                            onSubmitted: (_) => _saveConnectionSettings(),
                            decoration: const InputDecoration(
                              labelText: 'Server Host',
                              hintText: 'localhost or IP address',
                              border: OutlineInputBorder(),
                              isDense: true,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: TextField(
                            controller: TextEditingController(text: _serverPort.toString()),
                            onChanged: (v) => _serverPort = int.tryParse(v) ?? 8080,
                            onSubmitted: (_) => _saveConnectionSettings(),
                            decoration: const InputDecoration(
                              labelText: 'Port',
                              hintText: '8080',
                              border: OutlineInputBorder(),
                              isDense: true,
                            ),
                            keyboardType: TextInputType.number,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Status: ${_socketService.connectionInfo}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 24),

            // Status Display
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: isRec ? Colors.red.withOpacity(0.12) : 
                       _isConnected ? Colors.green.withOpacity(0.12) : Colors.white10,
                border: Border.all(
                  color: isRec ? Colors.redAccent : 
                         _isConnected ? Colors.greenAccent : Colors.white24,
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                children: [
                  Text(
                    isRec ? 'Recording & Streaming ${_fmt(_elapsed)}' : _status,
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      color: isRec ? Colors.redAccent : 
                             _isConnected ? Colors.greenAccent : Colors.white70,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  if (isRec) ...[
                    const SizedBox(height: 4),
                    Text(
                      '${_formatBytes(_bytesSent)} sent in $_packetsSpent packets',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.white60,
                      ),
                    ),
                  ],
                ],
              ),
            ),
            
            const SizedBox(height: 32),

            // Record Button
            ScaleTransition(
              scale: _pulse,
              child: SizedBox(
                width: 200,
                height: 200,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: isRec
                          ? [Colors.red.shade400, Colors.red.shade700]
                          : _isConnected
                              ? [Colors.indigo.shade400, Colors.indigo.shade700]
                              : [Colors.grey.shade600, Colors.grey.shade800],
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: (isRec ? Colors.red : 
                               _isConnected ? Colors.indigo : Colors.grey)
                            .withOpacity(0.45),
                        blurRadius: 32,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: Material(
                    color: Colors.transparent,
                    child: InkWell(
                      customBorder: const CircleBorder(),
                      onTap: _isConnected ? _toggleRecording : null,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            isRec ? Icons.stop_rounded : Icons.mic_rounded,
                            size: 64,
                            color: _isConnected ? Colors.white : Colors.white54,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            isRec ? 'Stop' : 
                            _isConnected ? 'Record' : 'Connect First',
                            style: TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.5,
                              color: _isConnected ? Colors.white : Colors.white54,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),

            const SizedBox(height: 32),

            // Help Text
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Quick Setup',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      '1. Start the Python server: python main.py --mobile-only\n'
                      '2. Update the server host to your computer\'s IP address\n'
                      '3. Tap the connection icon to connect\n'
                      '4. Press Record to start real-time audio streaming',
                      style: TextStyle(fontSize: 12, color: Colors.white70),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}