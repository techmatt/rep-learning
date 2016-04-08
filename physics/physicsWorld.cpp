
#include "main.h"

void PhysicsWorld::render(const PhysicsRender &render, ColorImageR8G8B8A8 &image)
{
    vector<vec3f> bodyColors;
    
    for (int i = 0; i < 4; i++)
        bodyColors.push_back(render.wallColor);
    bodyColors.push_back(render.ballColor);

    image.setPixels(vec4uc(0, 0, 0, 255));
    const float domain = 6.5f;
    for (auto &p : image)
    {
        vec2f pt;
        pt.x = math::linearMap(0.0f, image.getWidth() - 1.0f, domain, -domain, (float)p.x);
        pt.y = math::linearMap(0.0f, image.getHeight() - 1.0f, domain, -domain, (float)p.y);

        for (int body = 0; body < (int)bodies.size(); body++)
        {
            if (bodies[body].interior(pt))
            {
                p.value = ColorUtils::toColor8(bodyColors[body]);
            }
        }
    }
}

void PhysicsWorld::init()
{
    b2Vec2 gravity(0.0f, -100.0f);

    bodies.clear();
    world = new b2World(gravity);
    
    auto bboxFromLine = [](const vec2f &a, const vec2f &b, float radius)
    {
        bbox2f result;
        result.include(a - vec2f(radius, radius));
        result.include(a + vec2f(radius, radius));
        result.include(b - vec2f(radius, radius));
        result.include(b + vec2f(radius, radius));
        return result;
    };

    const float d = 7.0f;
    addStationaryBox(bboxFromLine(vec2f(-d, -d), vec2f(-d, d), 1.0f), vec3f(0.0f, 0.0f, 0.0f));
    addStationaryBox(bboxFromLine(vec2f(-d, d), vec2f(d, d), 1.0f), vec3f(0.0f, 0.0f, 0.0f));
    addStationaryBox(bboxFromLine(vec2f(d, d), vec2f(d, -d), 1.0f), vec3f(0.0f, 0.0f, 0.0f));
    addStationaryBox(bboxFromLine(vec2f(d, -d), vec2f(-d, -d), 1.0f), vec3f(0.0f, 0.0f, 0.0f));

    //bodies[0].body->ApplyForceToCenter(b2Vec2(0.0f, -300.0f), true);
    //bodies[1].body->ApplyForceToCenter(b2Vec2(0.0f, 1000.0f), true);

    const int sphereCount = 1;
    for (int sphereIndex = 0; sphereIndex < sphereCount; sphereIndex++)
    {
        auto r = []() { return util::randomUniform(-1.0f, 1.0f); };
        auto rv = [&]() { return vec2f(r(), r()); };
        addSphere(rv() * d * 0.8f, vec3f(1.0f, 1.0f, 1.0f), 1.0f, 1.0f, rv() * 6000.0f + 3000.0f);
    }
}

void PhysicsWorld::macroStep()
{
    microStep();
    microStep();
}

void PhysicsWorld::microStep()
{
    // Prepare for simulation. Typically we use a time step of 1/60 of a
    // second (60Hz) and 10 iterations. This provides a high quality simulation
    // in most game scenarios.
    float32 timeStep = 1.0f / 60.0f;
    int32 velocityIterations = 6;
    int32 positionIterations = 2;

    // Instruct the world to perform a single step of simulation.
    // It is generally best to keep the time step and iterations fixed.
    world->Step(timeStep, velocityIterations, positionIterations);
}

void PhysicsWorld::addSphere(const vec2f &position, const vec3f &color, float radius, float density, const vec2f &initialForce)
{
    // Define the dynamic body. We set its position and call the body factory.
    b2BodyDef bodyDef;
    bodyDef.type = b2_dynamicBody;
    bodyDef.position.Set(position.x, position.y);
    b2Body *body = world->CreateBody(&bodyDef);
    
    // Define another box shape for our dynamic body.
    b2CircleShape circleShape;

    circleShape.m_p.Set(0, 0); //position, relative to body position
    circleShape.m_radius = radius;
    
    // Define the dynamic body fixture.
    b2FixtureDef fixtureDef;
    fixtureDef.shape = &circleShape;

    // Set the box density to be non-zero, so it will be dynamic.
    fixtureDef.density = density;
    fixtureDef.friction = 0.3f;
    fixtureDef.restitution = 0.7f;

    // Add the shape to the body.
    body->CreateFixture(&fixtureDef);

    body->ApplyForceToCenter(b2Vec2(initialForce.x, initialForce.y), true);

    PhysicsBody pbody = PhysicsBody(body, color, BodyType::Circle);
    pbody.circleRadius = radius;

    bodies.push_back(pbody);
}

void PhysicsWorld::addStationaryBox(const bbox2f &box, const vec3f &color)
{
    // Define the dynamic body. We set its position and call the body factory.
    b2BodyDef bodyDef;
    bodyDef.type = b2_staticBody;
    bodyDef.position.Set(box.getCenter().x, box.getCenter().y);
    b2Body *body = world->CreateBody(&bodyDef);

    PhysicsBody pbody = PhysicsBody(body, vec3f(1.0f, 1.0f, 0.0f), BodyType::Box);
    pbody.absoluteBox = box;
    pbody.centeredBox = bbox2f(box.getMin() - box.getCenter(), box.getMax() - box.getCenter());

    // Define another box shape for our dynamic body.
    b2PolygonShape boxShape;

    b2Vec2 points[4];
    vec2f vmin = pbody.centeredBox.getMin();
    vec2f vmax = pbody.centeredBox.getMax();
    points[0] = b2Vec2(vmin.x, vmin.y);
    points[1] = b2Vec2(vmax.x, vmin.y);
    points[2] = b2Vec2(vmax.x, vmax.y);
    points[3] = b2Vec2(vmin.x, vmax.y);
    boxShape.Set(points, 4);

    // Define the dynamic body fixture.
    b2FixtureDef fixtureDef;
    fixtureDef.shape = &boxShape;

    fixtureDef.density = 0.0f;

    // Override the default friction.
    fixtureDef.friction = 0.3f;
    fixtureDef.restitution = 0.7f;

    // Add the shape to the body.
    body->CreateFixture(&fixtureDef);

    bodies.push_back(pbody);
}
