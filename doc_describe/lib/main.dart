import 'dart:async';
import 'dart:io';
// flutter run --release -d 00008101-000439922E12001E
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

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
  final AudioRecorder _recorder = AudioRecorder();
  bool _isRecording = false;
  String _status = 'Tap to record';
  String? _lastPath;

  Duration _elapsed = Duration.zero;
  Timer? _timer;

  late final AnimationController _pulse;

  // ---- Backend URL: set this for your target
  // iOS/macOS simulator:  http://127.0.0.1:8080/upload
  // Android emulator:     http://10.0.2.2:8080/upload
  // Real device:          http://<your-computer-LAN-IP>:8080/upload
  static const String _serverUrl = 'http://172.20.10.6:8080/upload';

  @override
  void initState() {
    super.initState();
    _pulse =
        AnimationController(
          vsync: this,
          duration: const Duration(milliseconds: 1400),
          lowerBound: 0.9,
          upperBound: 1.08,
        )..addStatusListener((s) {
          if (s == AnimationStatus.completed) _pulse.reverse();
          if (s == AnimationStatus.dismissed) _pulse.forward();
        });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _pulse.dispose();
    _recorder.dispose();
    super.dispose();
  }

  Future<Directory> _recordingsDir() async {
    final base = await getApplicationSupportDirectory();
    final dir = Directory(p.join(base.path, 'recordings'));
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  Future<void> _toggle() async =>
      _isRecording ? _stopRecording() : _startRecording();

  Future<void> _startRecording() async {
    try {
      final hasPerm = await _recorder.hasPermission();
      if (!hasPerm) {
        setState(() => _status = 'Microphone permission denied');
        return;
      }
    } catch (_) {}

    try {
      final dir = await _recordingsDir();
      final ts = DateTime.now().toIso8601String().replaceAll(':', '-');
      final path = p.join(dir.path, 'docdescribe_$ts.wav');

      const cfg = RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      );
      await _recorder.start(cfg, path: path);

      setState(() {
        _isRecording = true;
        _status = 'Recording…';
        _elapsed = Duration.zero;
        _lastPath = null;
      });

      _timer?.cancel();
      _timer = Timer.periodic(
        const Duration(seconds: 1),
        (_) => setState(() => _elapsed += const Duration(seconds: 1)),
      );
      _pulse.forward();
    } catch (e) {
      setState(() => _status = 'Failed to start: $e');
    }
  }

  Future<void> _stopRecording() async {
    try {
      final savedPath = await _recorder.stop();
      _timer?.cancel();
      _pulse.stop();

      setState(() {
        _isRecording = false;
        _status = 'Saved';
        _lastPath = savedPath;
      });

      if (savedPath != null) {
        await _uploadFile(savedPath);
      }
    } catch (e) {
      setState(() => _status = 'Failed to stop: $e');
    }
  }

  Future<void> _uploadFile(String path) async {
    try {
      setState(() => _status = 'Uploading…');
      final uri = Uri.parse(_serverUrl);
      final req = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', path));
      final resp = await req.send();
      final body = await resp.stream.bytesToString();

      if (!mounted) return;
      setState(
        () => _status = (resp.statusCode == 200)
            ? 'Uploaded ✓'
            : 'Upload failed: ${resp.statusCode} ${resp.reasonPhrase} • $body',
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = 'Upload error: $e');
    }
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
      appBar: AppBar(title: const Text('Doc Describe'), centerTitle: true),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // status line (small + clean)
            Padding(
              padding: const EdgeInsets.only(bottom: 24),
              child: Text(
                isRec ? 'Recording… ${_fmt(_elapsed)}' : _status,
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: Colors.white70,
                ),
              ),
            ),
            // mic button
            ScaleTransition(
              scale: _pulse,
              child: SizedBox(
                width: 180,
                height: 180,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: isRec
                          ? [Colors.red.shade400, Colors.red.shade700]
                          : [Colors.indigo.shade400, Colors.indigo.shade700],
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: (isRec ? Colors.red : Colors.indigo).withOpacity(
                          0.45,
                        ),
                        blurRadius: 28,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: Material(
                    color: Colors.transparent,
                    child: InkWell(
                      customBorder: const CircleBorder(),
                      onTap: _toggle,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            isRec ? Icons.stop_rounded : Icons.mic_rounded,
                            size: 58,
                            color: Colors.white,
                          ),
                          const SizedBox(height: 10),
                          Text(
                            isRec ? 'Stop' : 'Record',
                            style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.5,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
            if (_lastPath != null) ...[
              const SizedBox(height: 16),
              Text(
                'Saved:\n$_lastPath',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white54, fontSize: 12),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
