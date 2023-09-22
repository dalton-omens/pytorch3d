# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F

from ..common.datatypes import Device
from .utils import convert_to_tensors_and_broadcast, TensorProperties


def diffuse(normals, color, direction) -> torch.Tensor:
    """
    Calculate the diffuse component of light reflection using Lambert's
    cosine law.

    Args:
        normals: (N, ..., 3) xyz normal vectors. Normals and points are
            expected to have the same shape.
        color: (1, 3) or (N, 3) RGB color of the diffuse component of the light.
        direction: (x,y,z) direction of the light

    Returns:
        colors: (N, ..., 3), same shape as the input points.

    The normals and light direction should be in the same coordinate frame
    i.e. if the points have been transformed from world -> view space then
    the normals and direction should also be in view space.

    NOTE: to use with the packed vertices (i.e. no batch dimension) reformat the
    inputs in the following way.

    .. code-block:: python

        Args:
            normals: (P, 3)
            color: (N, 3)[batch_idx, :] -> (P, 3)
            direction: (N, 3)[batch_idx, :] -> (P, 3)

        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes, batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx()
        depending on whether points refers to the vertex coordinates or
        average/interpolated face coordinates.
    """
    # TODO: handle multiple directional lights per batch element.
    # TODO: handle attenuation.

    # Ensure color and location have same batch dimension as normals
    normals, color, direction = convert_to_tensors_and_broadcast(
        normals, color, direction, device=normals.device
    )

    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as normals. Assume first dim = batch dim and last dim = 3.
    points_dims = normals.shape[1:-1]
    expand_dims = (-1,) + (1,) * len(points_dims) + (3,)
    if direction.shape != normals.shape:
        direction = direction.view(expand_dims)
    if color.shape != normals.shape:
        color = color.view(expand_dims)

    # Renormalize the normals in case they have been interpolated.
    # We tried to replace the following with F.cosine_similarity, but it wasn't faster.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    angle = F.relu(torch.sum(normals * direction, dim=-1))
    return color * angle[..., None]


def specular(
    points, normals, direction, color, camera_position, shininess
) -> torch.Tensor:
    """
    Calculate the specular component of light reflection.

    Args:
        points: (N, ..., 3) xyz coordinates of the points.
        normals: (N, ..., 3) xyz normal vectors for each point.
        color: (N, 3) RGB color of the specular component of the light.
        direction: (N, 3) vector direction of the light.
        camera_position: (N, 3) The xyz position of the camera.
        shininess: (N)  The specular exponent of the material.

    Returns:
        colors: (N, ..., 3), same shape as the input points.

    The points, normals, camera_position, and direction should be in the same
    coordinate frame i.e. if the points have been transformed from
    world -> view space then the normals, camera_position, and light direction
    should also be in view space.

    To use with a batch of packed points reindex in the following way.
    .. code-block:: python::

        Args:
            points: (P, 3)
            normals: (P, 3)
            color: (N, 3)[batch_idx] -> (P, 3)
            direction: (N, 3)[batch_idx] -> (P, 3)
            camera_position: (N, 3)[batch_idx] -> (P, 3)
            shininess: (N)[batch_idx] -> (P)
        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx().
    """
    # TODO: handle multiple directional lights
    # TODO: attenuate based on inverse squared distance to the light source

    if points.shape != normals.shape:
        msg = "Expected points and normals to have the same shape: got %r, %r"
        raise ValueError(msg % (points.shape, normals.shape))

    # Ensure all inputs have same batch dimension as points
    matched_tensors = convert_to_tensors_and_broadcast(
        points, color, direction, camera_position, shininess, device=points.device
    )
    _, color, direction, camera_position, shininess = matched_tensors

    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as points. Assume first dim = batch dim and last dim = 3.
    points_dims = points.shape[1:-1]
    expand_dims = (-1,) + (1,) * len(points_dims)
    if direction.shape != normals.shape:
        direction = direction.view(expand_dims + (3,))
    if color.shape != normals.shape:
        color = color.view(expand_dims + (3,))
    if camera_position.shape != normals.shape:
        camera_position = camera_position.view(expand_dims + (3,))
    if shininess.shape != normals.shape:
        shininess = shininess.view(expand_dims)

    # Renormalize the normals in case they have been interpolated.
    # We tried a version that uses F.cosine_similarity instead of renormalizing,
    # but it was slower.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    cos_angle = torch.sum(normals * direction, dim=-1)
    # No specular highlights if angle is less than 0.
    mask = (cos_angle > 0).to(torch.float32)

    # Calculate the specular reflection.
    view_direction = camera_position - points
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
    reflect_direction = -direction + 2 * (cos_angle[..., None] * normals)

    # Cosine of the angle between the reflected light ray and the viewer
    alpha = F.relu(torch.sum(view_direction * reflect_direction, dim=-1)) * mask
    return color * torch.pow(alpha, shininess)[..., None]


class DirectionalLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        direction=((0, 1, 0),),
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component.
            diffuse_color: RGB color of the diffuse component.
            specular_color: RGB color of the specular component.
            direction: (x, y, z) direction vector of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            direction=direction,
        )
        _validate_light_properties(self)
        if self.direction.shape[-1] != 3:
            msg = "Expected direction to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.direction.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points=None) -> torch.Tensor:
        # NOTE: Points is not used but is kept in the args so that the API is
        # the same for directional and point lights. The call sites should not
        # need to know the light type.
        return diffuse(
            normals=normals,
            color=self.diffuse_color,
            direction=self.direction,
        )

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=self.direction,
            camera_position=camera_position,
            shininess=shininess,
        )


class PointLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        location=((0, 1, 0),),
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            location=location,
        )
        _validate_light_properties(self)
        if self.location.shape[-1] != 3:
            msg = "Expected location to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.location.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def reshape_location(self, points) -> torch.Tensor:
        """
        Reshape the location tensor to have dimensions
        compatible with the points which can either be of
        shape (P, 3) or (N, H, W, K, 3).
        """
        if self.location.ndim == points.ndim:
            return self.location
        return self.location[:, None, None, None, :]

    def diffuse(self, normals, points) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=direction,
            camera_position=camera_position,
            shininess=shininess,
        )


class AmbientLights(TensorProperties):
    """
    A light object representing the same color of light everywhere.
    By default, this is white, which effectively means lighting is
    not used in rendering.

    Unlike other lights this supports an arbitrary number of channels, not just 3 for RGB.
    The ambient_color input determines the number of channels.
    """

    def __init__(self, *, ambient_color=None, device: Device = "cpu") -> None:
        """
        If ambient_color is provided, it should be a sequence of
        triples of floats.

        Args:
            ambient_color: RGB color
            device: Device (as str or torch.device) on which the tensors should be located

        The ambient_color if provided, should be
            - tuple/list of C-element tuples of floats
            - torch tensor of shape (1, C)
            - torch tensor of shape (N, C)
        where C is the number of channels and N is batch size.
        For RGB, C is 3.
        """
        if ambient_color is None:
            ambient_color = ((1.0, 1.0, 1.0),)
        super().__init__(ambient_color=ambient_color, device=device)

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        return self._zeros_channels(points)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return self._zeros_channels(points)

    def _zeros_channels(self, points: torch.Tensor) -> torch.Tensor:
        ch = self.ambient_color.shape[-1]
        return torch.zeros(*points.shape[:-1], ch, device=points.device)


class MultiPointLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_colors=(((0.3, 0.3, 0.3),),),
        specular_colors=(((0.2, 0.2, 0.2),),),
        locations=(((0, 1, 0),),),
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_colors: RGB color of the diffuse component for each light
            specular_colors: RGB color of the specular component for each light
            locations: xyz positions of each light
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, num_lights, 3)
            - torch tensor of shape (N, num_lights, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        ambient_color is (1, 3) or (N, 3) because there is only one ambient color.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_colors=diffuse_colors,
            specular_colors=specular_colors,
            locations=locations,
        )
        props = ("ambient_color", "diffuse_colors", "specular_colors")
        for n in props:
            t = getattr(self, n)
            if t.shape[-1] != 3:
                msg = "Expected %s to have shape (N, 3); got %r"
                raise ValueError(msg % (n, t.shape))
        if self.locations.shape[-1] != 3:
            msg = "Expected locations to have shape (N, num_lights, 3); got %r"
            raise ValueError(msg % repr(self.locations.shape))
        if self.diffuse_colors.ndim != 3:
            msg = "Expected diffuse_colors to have shape (N, num_lights, 3); got %r"
            raise ValueError(msg % repr(self.diffuse_colors.shape))
        if self.specular_colors.ndim != 3:
            msg = "Expected specular_colors to have shape (N, num_lights, 3); got %r"
            raise ValueError(msg % repr(self.specular_colors.shape))
        if self.diffuse_colors.shape[-2] != self.specular_colors.shape[-2] or \
           self.diffuse_colors.shape[-2] != self.locations.shape[-2]:
            msg = "Mismatched num_lights dimension for inputs:\n" + \
              f"diffuse_colors {diffuse_colors.shape}\nspecular_colors {specular_colors.shape}" + \
              f"\nlocations {locations.shape}"
            raise ValueError(msg)

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def reshape_location(self, points) -> torch.Tensor:
        """
        Reshape the location tensors to have dimensions
        compatible with the points which can either be of
        shape (P, 3) or (N, H, W, K, 3).
        """
        if self.locations.ndim - 1 == points.ndim:
            # pyre-fixme[7]
            return self.locations
        # pyre-fixme[29]
        return self.locations[:, None, None, None, :, :]

    def diffuse(self, normals, points) -> torch.Tensor:
        locations = self.reshape_location(points)
        diffuse_sum = torch.zeros_like(normals)
        for i in range(locations.shape[-2]):
            location = locations[..., i, :]
            direction = location - points
            color = self.diffuse_colors[..., i, :]
            diffuse_sum += diffuse(normals=normals,
                                   color=color,
                                   direction=direction)
        return diffuse_sum

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        locations = self.reshape_location(points)
        specular_sum = torch.zeros_like(normals)
        for i in range(locations.shape[-2]):
            location = locations[..., i, :]
            direction = location - points
            color = self.specular_colors[..., i, :]
            specular_sum += specular(
                points=points,
                normals=normals,
                color=color,
                direction=direction,
                camera_position=camera_position,
                shininess=shininess,
            )
        return specular_sum


class SphericalHarmonicsLights(TensorProperties):
    """
    Spherical harmonics lighting for representing an environment light.
    Depends on the normal direction, and not the view direction.
    Ramamoorthi, et al. "An Efficient Representation for Irradiance Environment Maps" (2001)
    """
    def __init__(self, sh_params, device: Device = "cpu") -> None:
        """
        Args:
            sh_params: (N, 9, 3) Parameters for first 3 bands of SH. The order of parameters is 
                       [L0 L1-1 L10 L11 L2-2 L2-1 L20 L21 L22]
            device: Device (as str or torch.device) on which the tensors should be located
        """

        '''        
        "Grace" SH params
        [[0.7953949,  0.4405923,  0.5459412], # 1 (L00)
        [0.3981450,  0.3526911,  0.6097158],  # Y (L1-1)
        [-0.3424573, -0.1838151, -0.2715583], # Z (L10)
        [-0.2944621, -0.0560606,  0.0095193], # X (L11)
        [-0.1123051, -0.0513088, -0.1232869], # YX (L2-2)
        [-0.2645007, -0.2257996, -0.4785847], # YZ (L2-1)
        [-0.1569444, -0.0954703, -0.1485053], # 3Z^2 - 1 (L20)
        [0.5646247,  0.2161586,  0.1402643],  # XZ (L21)
        [0.2137442, -0.0547578, -0.3061700]]  # X^2 - Y^2 (L22)
        '''
        super().__init__(device=device, ambient_color=((0.0, 0.0, 0.0),), sh_params=sh_params)

        if len(self.sh_params.shape) != 3 or self.sh_params.shape[-2:] != (9, 3):
            raise ValueError(f"Expected sh_params to have shape (N, 9, 3); got {self.sh_params.shape}")

        self.c1 = 0.429043
        self.c2 = 0.511664
        self.c3 = 0.743125
        self.c4 = 0.886227
        self.c5 = 0.247708

    def clone(self):
        other = self.__class__(device=self.device)
        copy = super().clone(other)
        copy.c1 = self.c1
        copy.c2 = self.c2
        copy.c3 = self.c3
        copy.c4 = self.c4
        copy.c5 = self.c5
        return copy

    def diffuse(self, normals, points) -> torch.Tensor:
        # normals: (B, ..., 3)
        input_shape = normals.shape
        B = input_shape[0]

        normals = normals.view(B, -1, 3, 1)
        x, y, z = normals.unbind(-2)
        sh_params = self.sh_params.view(-1, 9, 1, 3)
        color = (
        self.c4 * sh_params[:, 0] +                                          # 1         (L00)
        2.0 * self.c2 * sh_params[:, 1] * y +                                # Y         (L1-1)
        2.0 * self.c2 * sh_params[:, 2]  * z +                               # Z         (L10)
        2.0 * self.c2 * sh_params[:, 3]  * x +                               # X         (L11)
        2.0 * self.c1 * sh_params[:, 4] * x * y +                            # YX        (L2-2)
        2.0 * self.c1 * sh_params[:, 5] * y * z +                            # YZ        (L2-1)
        (self.c3 * sh_params[:, 6] * z * z) - (self.c5 * sh_params[:, 6]) +  # 3Z^2 - 1  (L20)
        2.0 * self.c1 * sh_params[:, 7]  * x * z +                           # XZ        (L21)
        self.c1 * sh_params[:, 8] * (x * x - y * y)                          # X^2 - Y^2 (L22)
        )
        color = color.view(B, *input_shape[1:-1], 3)

        # return torch.clip(color, 0, 1)
        return color

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return torch.zeros_like(points)

def _validate_light_properties(obj) -> None:
    props = ("ambient_color", "diffuse_color", "specular_color")
    for n in props:
        t = getattr(obj, n)
        if t.shape[-1] != 3:
            msg = "Expected %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
