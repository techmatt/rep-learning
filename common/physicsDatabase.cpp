
#include "main.h"

const int physicsHistoryWidth = 64;
const int physicsHistoryHeight = 64;

const int physicsFutureWidth = 64;
const int physicsFutureHeight = 64;

void PhysicsNetDatabase::init()
{
    
}

PhysicsNetEntry PhysicsNetDatabase::makeRandomEntry()
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

    PhysicsNetEntry result;
    result.history.allocate(physicsHistoryWidth, physicsHistoryHeight, PhysicsNetEntry::historyFrameCount * 3);

    for (int i = 0; i < PhysicsNetEntry::historyFrameCount; i++)
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

    result.futureFrames.resize(PhysicsNetEntry::futureFrameCount);
    for (int i = 0; i < max(PhysicsNetEntry::futureFrameCount, PhysicsNetEntry::futureStateCount); i++)
    {
        if (i < PhysicsNetEntry::futureFrameCount)
        {
            ColorImageR8G8B8A8 image(physicsFutureWidth, physicsFutureHeight);
            world.render(renderParams, image);
            result.futureFrames[i] = makeGrid3(image);
        }
        if (i < PhysicsNetEntry::futureStateCount)
        {
            result.futureStates.push_back(world.getState());
        }

        world.macroStep();
    }
    return result;
}

void PhysicsNetDatabase::createDatabase(const string &directory, int sampleCount)
{
    mBase::Writer<PhysicsNetEntry> writer(directory);

    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        if (sampleIndex % 100 == 0)
            cout << "Sample " << sampleIndex << " / " << sampleCount << endl;

        PhysicsNetEntry sample;

        sample = makeRandomEntry();

        if (sampleIndex <= 2)
        {
            sample.save("test" + to_string(sampleIndex));
        }

        writer.addRecord(sample);
    }

    writer.finalize();
}
