The folder glm-git-modified/ contains a slightly modified version of glm (https://github.com/g-truc/glm), cloned from git and based off the pre-version of 0.9.8.0.

The modifications fix a problem that causes non-trivially-copyable types not be allowed as type T in glm's vector, matrix and quaternion types.
A pull request has been proposed to glm: https://github.com/g-truc/glm/pull/543

My modified version is available at https://github.com/patrikhuber/glm. We stay as close as possible to the original version and try to get these fixes into upstream glm.
