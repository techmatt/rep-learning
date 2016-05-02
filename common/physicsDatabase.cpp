
#include "main.h"

const int physicsHistoryWidth = 64;
const int physicsHistoryHeight = 64;

const int physicsFutureWidth = 64;
const int physicsFutureHeight = 64;

void PhysicsNetDatabase::init()
{
    
}

PhysicsNetEntryImage PhysicsNetDatabase::makeRandomEntryImage()
{
    auto makeGrid3 = [](const ColorImageR8G8B8A8 &image)
    {
        Grid3uc g(image.getWidth(), image.getHeight(), 3);
        for (auto &v : g)
        {
            v.value = image(v.x, v.y)[(int)v.z];
        }
        return g;
    };

    PhysicsWorld world;
    world.init();
    
    const int stepCount = util::randomInteger(10, 50);
    for (int i = 0; i < stepCount; i++)
    {
        world.macroStep();
    }

    const PhysicsRender renderParams = PhysicsRender::random();

    PhysicsNetEntryImage result;
    result.history.allocate(physicsHistoryWidth, physicsHistoryHeight, PhysicsNetEntryImage::historyFrameCount * 3);

    for (int i = 0; i < PhysicsNetEntryImage::historyFrameCount; i++)
    {
        ColorImageR8G8B8A8 image(physicsHistoryWidth, physicsHistoryHeight);
        world.render(renderParams, image);
        for (int y = 0; y < physicsHistoryHeight; y++)
            for (int x = 0; x < physicsHistoryWidth; x++)
                for (int c = 0; c < 3; c++)
                {
                    result.history(x, y, i * 3 + c) = image(x, y)[c];
                }

        world.macroStep();
    }

    result.futureFrames.resize(PhysicsNetEntryImage::futureFrameCount);
    for (int i = 0; i < max(PhysicsNetEntryImage::futureFrameCount, PhysicsNetEntryImage::futureStateCount); i++)
    {
        if (i < PhysicsNetEntryImage::futureFrameCount)
        {
            ColorImageR8G8B8A8 image(physicsFutureWidth, physicsFutureHeight);
            world.render(renderParams, image);
            result.futureFrames[i] = makeGrid3(image);
        }
        if (i < PhysicsNetEntryImage::futureStateCount)
        {
            result.futureStates.push_back(world.getState());
        }

        world.macroStep();
    }
    return result;
}

PhysicsNetEntryFlat PhysicsNetDatabase::makeRandomEntryFlat()
{
    PhysicsWorld world;
    world.init();

    const int stepCount = util::randomInteger(10, 50);
    for (int i = 0; i < stepCount; i++)
    {
        world.macroStep();
    }

    PhysicsNetEntryFlat result;

    result.states.resize(PhysicsNetEntryFlat::stateCount);
    for (int i = 0; i < PhysicsNetEntryFlat::stateCount; i++)
    {
        vector<float> startState = world.getState();
        world.macroStep();
        vector<float> endState = world.getState();

        vector<float> fullState = startState;
        fullState.push_back(endState[0] - startState[0]);
        fullState.push_back(endState[1] - startState[1]);
        result.states.push_back(fullState);
    }
    return result;
}

void PhysicsNetDatabase::createDatabaseImage(const string &directory, int sampleCount)
{
    mBase::Writer<PhysicsNetEntryImage> writer(directory);

    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        if (sampleIndex % 100 == 0)
            cout << "Sample " << sampleIndex << " / " << sampleCount << endl;

        PhysicsNetEntryImage sample = makeRandomEntryImage();

        if (sampleIndex <= 2)
        {
            sample.save("test" + to_string(sampleIndex));
        }

        writer.addRecord(sample);
    }

    writer.finalize();
}


void PhysicsNetDatabase::createDatabaseFlat(const string &filename, int sampleCount)
{
    ofstream file(filename);
    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        if (sampleIndex % 100 == 0)
            cout << "Sample " << sampleIndex << " / " << sampleCount << endl;

        PhysicsNetEntryFlat sample = makeRandomEntryFlat();
        for (auto &s : sample.states)
            for (auto &v : s)
            {
                file << v << ",";
            }
        file << endl;
    }
}