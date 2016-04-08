
enum class BodyType
{
    Circle,
    Box,
};

struct PhysicsRender
{
    static PhysicsRender random()
    {
        PhysicsRender r;
        r.wallColor = vec3f(1.0f, 0.0f, 0.0f);
        r.ballColor = vec3f(0.0f, 0.0f, 1.0f);
        return r;
    }
    vec3f wallColor;
    vec3f ballColor;
};

struct PhysicsBody
{
    PhysicsBody()
    {
        body = nullptr;
    }

    PhysicsBody(b2Body *_body, const vec3f &_color, BodyType _type)
    {
        color = _color;
        body = _body;
        type = _type;
    }

    vec3f position() const
    {
        return vec3f(body->GetPosition().x, body->GetPosition().y, 0.0f);
    }

    float angle() const
    {
        return body->GetAngle();
    }

    bool interior(const vec2f &point) const
    {
        if (type == BodyType::Box)
        {
            return absoluteBox.intersects(point);
        }
        if (type == BodyType::Circle)
        {
            const float eps = 0.1f;
            return vec2f::distSq(position().getVec2(), point) < math::square(circleRadius + eps);
        }
        return false;
    }

    BodyType type;
    vec3f color;
    b2Body *body;

    bbox2f absoluteBox;
    bbox2f centeredBox;
    float circleRadius;
};

struct PhysicsWorld
{
    void init();
    void microStep();
    void macroStep();
    void render(const PhysicsRender &render, ColorImageR8G8B8A8 &image);

    b2World *world;

    //b2Body *groundBody;
    
    vector<PhysicsBody> bodies;

private:
    void addSphere(const vec2f &position, const vec3f &color, float radius, float density, const vec2f &initialForce);
    void addBox(const bbox2f &box, const vec3f &color, float density);
    void addStationaryBox(const bbox2f &box, const vec3f &color);
    void updateBodies();
};

