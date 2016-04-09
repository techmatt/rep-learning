
struct PhysicsNetEntry
{
    static const int historyFrameCount = 4;
    static const int futureFrameCount = 1;
    static const int futureStateCount = 20;

    vector<ColorImageR8G8B8A8> makeVisualization()
    {
        auto extractHistoryFrame = [&](int index)
        {
            Grid3uc result(history.getDimX(), history.getDimY(), 3);
            for (auto &v : result)
            {
                v.value = history(v.x, v.y, index * 3 + v.z);
            }
            return result;
        };

        auto makeImage = [](const Grid3uc &g)
        {
            ColorImageR8G8B8A8 image((int)g.getDimX(), (int)g.getDimY(), vec4uc(0, 0, 0, 255));
            for (auto &v : g)
            {
                image((int)v.x, (int)v.y)[(int)v.z] = v.value;
            }
            return image;
        };

        vector<ColorImageR8G8B8A8> result;
        for (int i = 0; i < historyFrameCount; i++)
        {
            const Grid3uc frame = extractHistoryFrame(i);
            result.push_back(makeImage(frame));
        }

        for (int i = 0; i < futureFrameCount; i++)
        {
            result.push_back(makeImage(futureFrames[i]));
        }
        return result;
    }

    void save(const string &prefix)
    {
        const vector<ColorImageR8G8B8A8> images = makeVisualization();
        for (int i = 0; i < historyFrameCount; i++)
        {
            LodePNG::save(images[i], prefix + "_h" + to_string(i) + ".png");
        }

        for (int i = 0; i < futureFrameCount; i++)
        {
            LodePNG::save(images[i + historyFrameCount], prefix + "_n" + to_string(i) + ".png");
        }
    }

    // 128x128
    // frame 0 - 3 = history, 4 - 6 = future
    Grid3uc history;
    vector<Grid3uc> futureFrames;
    vector< vector<float> > futureStates;
};

template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const PhysicsNetEntry &data) {
    s.writePrimitive(data.history);
    for (int i = 0; i < PhysicsNetEntry::futureFrameCount; i++)
        s.writePrimitive(data.futureFrames[i]);
    s << data.futureStates;
    return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, PhysicsNetEntry &data) {
    s.readPrimitive(data.history);

    data.future.resize(PhysicsNetEntry::futureFrameCount);
    for (int i = 0; i < PhysicsNetEntry::futureFrameCount; i++)
        s.readPrimitive(data.futureFrames[i]);
    s >> data.futureStates;
    return s;
}

struct PhysicsNetDatabase
{
    void init();

    static PhysicsNetEntry makeRandomEntry();
    
    void createDatabase(const string &directory, int sampleCount);
};
